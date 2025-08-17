from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dataset import save_emb

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32).to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids, time_deltas, seq_len):
        
        combined_pos = position_ids.float() + torch.log(time_deltas.float() + 1.0) * 0.1

        freqs = torch.outer(combined_pos.flatten(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos().view(position_ids.shape[0], seq_len, self.dim)
        sin = emb.sin().view(position_ids.shape[0], seq_len, self.dim)
        return cos, sin

class HSTULayer(nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super().__init__()
        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.dropout_rate = dropout_rate

        self.uvqk_proj = nn.Linear(hidden_units, 2 * hidden_units + 2 * hidden_units)
        self.gating_norm = RMSNorm(hidden_units)
        self.output_proj = nn.Linear(hidden_units, hidden_units)
        self.output_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, cos, sin, attn_mask=None):
        batch_size, seq_len, _ = x.shape
        residual = x
        
        uvqk = self.uvqk_proj(x)
        u, v, q, k = torch.split(
            uvqk,
            [self.hidden_units, self.hidden_units, self.hidden_units, self.hidden_units],
            dim=-1,
        )
        
        u = F.silu(u)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        cos = cos.unsqueeze(2).repeat(1, 1, self.num_heads, 1)
        sin = sin.unsqueeze(2).repeat(1, 1, self.num_heads, 1)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        attn_weights = F.silu(scores)
        
        if attn_mask is not None:
            attn_weights = attn_weights * attn_mask.unsqueeze(1)
            
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)
        
        normed_attn_output = self.gating_norm(attn_output)
        gated_output = normed_attn_output * u
        
        transformed_output = self.output_proj(gated_output)
        transformed_output = self.output_dropout(transformed_output)
        
        output = residual + transformed_output
        return output

class BaselineModel(torch.nn.Module):
    def __init__(self, user_num, item_num, feat_statistics, feat_types, args):
        super(BaselineModel, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.maxlen = args.maxlen
        self.num_heads = args.num_heads
        self.temp = 0.07
        
        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num + 1, args.hidden_units, padding_idx=0)
        
        self.rotary_emb = RotaryEmbedding(dim=args.hidden_units // args.num_heads, device=self.dev)
        
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.sparse_emb = torch.nn.ModuleDict()
        self.emb_transform = torch.nn.ModuleDict()
        self._init_feat_info(feat_statistics, feat_types)
        userdim = args.hidden_units * (len(self.USER_SPARSE_FEAT) + 1 + len(self.USER_ARRAY_FEAT)) + len(
            self.USER_CONTINUAL_FEAT
        )
        itemdim = (
            args.hidden_units * (len(self.ITEM_SPARSE_FEAT) + 1 + len(self.ITEM_ARRAY_FEAT))
            + len(self.ITEM_CONTINUAL_FEAT)
            + args.hidden_units * len(self.ITEM_EMB_FEAT)
        )
        self.userdnn = torch.nn.Linear(userdim, args.hidden_units)
        self.itemdnn = torch.nn.Linear(itemdim, args.hidden_units)
        self.feat_fusion_norm = RMSNorm(args.hidden_units)
        self.input_norm = RMSNorm(args.hidden_units)

        self.hstu_layers = nn.ModuleList(
            [HSTULayer(args.hidden_units, args.num_heads, args.dropout_rate) for _ in range(args.num_blocks)]
        )
        self.last_layernorm = RMSNorm(args.hidden_units, eps=1e-6)
        for k in self.USER_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.USER_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_EMB_FEAT:
            self.emb_transform[k] = torch.nn.Linear(self.ITEM_EMB_FEAT[k], args.hidden_units)

    def _init_feat_info(self, feat_statistics, feat_types):
        self.USER_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['user_sparse']}
        self.USER_CONTINUAL_FEAT = feat_types['user_continual']
        self.ITEM_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['item_sparse']}
        self.ITEM_CONTINUAL_FEAT = feat_types['item_continual']
        self.USER_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['user_array']}
        self.ITEM_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['item_array']}
        EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in feat_types['item_emb']}

    def feat2tensor(self, seq_feature, k):
        batch_size = len(seq_feature)
        if k in self.ITEM_ARRAY_FEAT or k in self.USER_ARRAY_FEAT:
            max_array_len = 0
            max_seq_len = 0
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                max_seq_len = max(max_seq_len, len(seq_data))
                max_array_len = max(max_array_len, max(len(item_data) for item_data in seq_data))
            batch_data = np.zeros((batch_size, max_seq_len, max_array_len), dtype=np.int64)
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                for j, item_data in enumerate(seq_data):
                    actual_len = min(len(item_data), max_array_len)
                    batch_data[i, j, :actual_len] = item_data[:actual_len]
            return torch.from_numpy(batch_data).to(self.dev)
        else:
            max_seq_len = max(len(seq_feature[i]) for i in range(batch_size))
            batch_data = np.zeros((batch_size, max_seq_len), dtype=np.int64)
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                batch_data[i] = seq_data
            return torch.from_numpy(batch_data).to(self.dev)

    def feat2emb(self, seq, feature_array, mask=None, include_user=False):
        seq = seq.to(self.dev)
        if include_user:
            user_mask = (mask == 2).to(self.dev)
            item_mask = (mask == 1).to(self.dev)
            user_embedding = self.user_emb(user_mask * seq)
            item_embedding = self.item_emb(item_mask * seq)
            item_feat_list = [item_embedding]
            user_feat_list = [user_embedding]
        else:
            item_embedding = self.item_emb(seq)
            item_feat_list = [item_embedding]
        all_feat_types = [
            (self.ITEM_SPARSE_FEAT, 'item_sparse', item_feat_list),
            (self.ITEM_ARRAY_FEAT, 'item_array', item_feat_list),
            (self.ITEM_CONTINUAL_FEAT, 'item_continual', item_feat_list),
        ]
        if include_user:
            all_feat_types.extend(
                [
                    (self.USER_SPARSE_FEAT, 'user_sparse', user_feat_list),
                    (self.USER_ARRAY_FEAT, 'user_array', user_feat_list),
                    (self.USER_CONTINUAL_FEAT, 'user_continual', user_feat_list),
                ]
            )
        for feat_dict, feat_type, feat_list in all_feat_types:
            if not feat_dict: continue
            for k in feat_dict:
                tensor_feature = self.feat2tensor(feature_array, k)
                if feat_type.endswith('sparse'):
                    feat_list.append(self.sparse_emb[k](tensor_feature))
                elif feat_type.endswith('array'):
                    feat_list.append(self.sparse_emb[k](tensor_feature).sum(2))
                elif feat_type.endswith('continual'):
                    feat_list.append(tensor_feature.unsqueeze(2))
        for k in self.ITEM_EMB_FEAT:
            batch_size = len(feature_array)
            emb_dim = self.ITEM_EMB_FEAT[k]
            seq_len = len(feature_array[0])
            batch_emb_data = np.zeros((batch_size, seq_len, emb_dim), dtype=np.float32)
            for i, seq_item in enumerate(feature_array):
                for j, item in enumerate(seq_item):
                    if k in item:
                        batch_emb_data[i, j] = item[k]
            tensor_feature = torch.from_numpy(batch_emb_data).to(self.dev)
            item_feat_list.append(self.emb_transform[k](tensor_feature))
        all_item_emb = torch.cat(item_feat_list, dim=2)
        all_item_emb = torch.relu(self.itemdnn(all_item_emb))
        if include_user:
            all_user_emb = torch.cat(user_feat_list, dim=2)
            all_user_emb = torch.relu(self.userdnn(all_user_emb))
            seqs_emb = all_item_emb + all_user_emb
        else:
            seqs_emb = all_item_emb
        return seqs_emb

    def log2feats(self, log_seqs, mask, seq_feature, seq_timestamps):
        seqs = self.feat2emb(log_seqs, seq_feature, mask=mask, include_user=True)
        seqs = self.feat_fusion_norm(seqs)
        seqs *= self.item_emb.embedding_dim**0.5
        
        seqs = self.input_norm(seqs)
        seqs = self.emb_dropout(seqs)
        
        batch_size, maxlen, _ = seqs.shape
        
        position_ids = torch.arange(maxlen, device=self.dev).view(1, -1).expand(batch_size, -1)
        
        valid_ts_mask = (seq_timestamps != 0)
        first_ts = (seq_timestamps + (~valid_ts_mask * 1e10)).min(dim=1, keepdim=True)[0]
        time_deltas = torch.clamp(seq_timestamps - first_ts, min=0) * valid_ts_mask

        cos, sin = self.rotary_emb(position_ids, time_deltas, maxlen)
        cos = cos.to(seqs.dtype)
        sin = sin.to(seqs.dtype)
        
        attention_mask_pad = (mask.to(self.dev) != 0)
        causal_mask = torch.tril(torch.ones((maxlen, maxlen), dtype=torch.bool, device=self.dev))
        attention_mask = causal_mask & attention_mask_pad.unsqueeze(1)
        
        for layer in self.hstu_layers:
            seqs = layer(seqs, cos=cos, sin=sin, attn_mask=attention_mask)
            
        log_feats = self.last_layernorm(seqs)
        return log_feats

    def compute_infonce_loss(self, seq_embs, pos_embs, neg_embs, loss_mask):
        hidden_size = neg_embs.size(-1)
        seq_embs_norm = seq_embs / seq_embs.norm(dim=-1, keepdim=True)
        pos_embs_norm = pos_embs / pos_embs.norm(dim=-1, keepdim=True)
        neg_embs_norm = neg_embs / neg_embs.norm(dim=-1, keepdim=True)
        pos_logits = F.cosine_similarity(seq_embs_norm, pos_embs_norm, dim=-1).unsqueeze(-1)
        neg_embedding_all = neg_embs_norm.reshape(-1, hidden_size)
        neg_logits = torch.matmul(seq_embs_norm, neg_embedding_all.transpose(-1, -2))
        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        logits = logits[loss_mask.bool()] / self.temp
        if logits.size(0) == 0:
            return torch.tensor(0.0, device=self.dev)
        labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.int64)
        loss = F.cross_entropy(logits, labels)
        return loss

    def forward(self, user_item, pos_seqs, neg_seqs, mask, next_mask, next_action_type, seq_feature, pos_feature, neg_feature, seq_timestamps):
        loss_mask = (next_mask == 1)
        if not loss_mask.any():
            return torch.tensor(0.0, device=self.dev, requires_grad=True)
        log_feats = self.log2feats(user_item, mask, seq_feature, seq_timestamps)
        pos_embs = self.feat2emb(pos_seqs, pos_feature, include_user=False)
        neg_embs = self.feat2emb(neg_seqs, neg_feature, include_user=False)
        infonce_loss = self.compute_infonce_loss(log_feats, pos_embs, neg_embs, loss_mask)
        return infonce_loss

    def predict(self, log_seqs, seq_feature, mask, seq_timestamps):
        log_feats = self.log2feats(log_seqs, mask, seq_feature, seq_timestamps)
        final_feat = log_feats[:, -1, :]
        return final_feat

    def save_item_emb(self, item_ids, retrieval_ids, feat_dict, save_path, batch_size=1024):
        all_embs = []
        for start_idx in tqdm(range(0, len(item_ids), batch_size), desc="Saving item embeddings"):
            end_idx = min(start_idx + batch_size, len(item_ids))
            item_seq = torch.tensor(item_ids[start_idx:end_idx], device=self.dev).unsqueeze(0)
            batch_feat = []
            for i in range(start_idx, end_idx):
                batch_feat.append(feat_dict[i])
            batch_feat = np.array(batch_feat, dtype=object)
            batch_emb = self.feat2emb(item_seq, [batch_feat], include_user=False).squeeze(0)
            all_embs.append(batch_emb.detach().cpu().numpy().astype(np.float32))
        final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
        final_embs = np.concatenate(all_embs, axis=0)
        save_emb(final_embs, Path(save_path, 'embedding.fbin'))
        save_emb(final_ids, Path(save_path, 'id.u64bin'))