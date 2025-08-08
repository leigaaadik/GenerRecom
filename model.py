# file: model.py

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dataset import save_emb

class FlashMultiHeadAttention(nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super(FlashMultiHeadAttention, self).__init__()
        self.hidden_units, self.num_heads, self.dropout_rate = hidden_units, num_heads, dropout_rate
        self.head_dim = hidden_units // num_heads
        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"
        self.q_linear, self.k_linear, self.v_linear, self.out_linear = [nn.Linear(hidden_units, hidden_units) for _ in range(4)]

    def forward(self, query, key, value, attn_mask=None):
        batch_size, seq_len, _ = query.size()
        Q, K, V = self.q_linear(query), self.k_linear(key), self.v_linear(value)
        Q, K, V = [x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) for x in (Q, K, V)]
        
        if hasattr(F, 'scaled_dot_product_attention'):
            if attn_mask is not None:
                attn_mask = attn_mask.unsqueeze(1)
            attn_output = F.scaled_dot_product_attention(Q, K, V, dropout_p=self.dropout_rate if self.training else 0.0, attn_mask=attn_mask)
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)
            return self.out_linear(attn_output), None
        else:
            scores = torch.matmul(Q, K.transpose(-2, -1)) * (self.head_dim ** -0.5)
            if attn_mask is not None:
                scores.masked_fill_(attn_mask.unsqueeze(1).logical_not(), float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training)
            attn_output = torch.matmul(attn_weights, V)
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)
            return self.out_linear(attn_output), None

class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1, self.dropout1, self.relu, self.conv2, self.dropout2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1), nn.Dropout(p=dropout_rate), nn.ReLU(), nn.Conv1d(hidden_units, hidden_units, kernel_size=1), nn.Dropout(p=dropout_rate)
    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        return outputs.transpose(-1, -2)

class RQEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_channels: list, latent_dim: int):
        super().__init__()
        self.stages = nn.ModuleList()
        in_dim = input_dim
        for out_dim in hidden_channels:
            self.stages.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU()))
            in_dim = out_dim
        self.stages.append(nn.Linear(in_dim, latent_dim))
    def forward(self, x):
        for stage in self.stages: x = stage(x)
        return x

class RQDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_channels: list, output_dim: int):
        super().__init__()
        self.stages = nn.ModuleList()
        in_dim = latent_dim
        for out_dim in hidden_channels:
            self.stages.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU()))
            in_dim = out_dim
        self.stages.append(nn.Linear(in_dim, output_dim))
    def forward(self, x):
        for stage in self.stages: x = stage(x)
        return x

class GatedFusion(nn.Module):
    def __init__(self, num_modalities, hidden_units):
        super().__init__()
        self.gate_layer = nn.Linear(num_modalities * hidden_units, num_modalities)
    def forward(self, modality_embs: list):
        if not modality_embs: return None
        if len(modality_embs) == 1: return modality_embs[0]
        gate_input = torch.cat(modality_embs, dim=-1)
        gate_values = F.softmax(self.gate_layer(gate_input), dim=-1).unsqueeze(-2)
        stacked_embs = torch.stack(modality_embs, dim=-1)
        return (stacked_embs * gate_values).sum(dim=-1)

class GenerativeFeatureSASRec(nn.Module):
    def __init__(self, user_num, item_num, feat_statistics, feat_types, args):
        super().__init__()
        self.user_num, self.item_num, self.dev, self.args = user_num, item_num, args.device, args
        self._init_feat_info(feat_statistics, feat_types)

        self.item_emb = nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.user_emb = nn.Embedding(self.user_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(args.maxlen + 1, args.hidden_units)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)
        
        all_sparse_feats = {**self.USER_SPARSE_FEAT, **self.ITEM_SPARSE_FEAT, **self.USER_ARRAY_FEAT, **self.ITEM_ARRAY_FEAT}
        self.sparse_emb = nn.ModuleDict({k: nn.Embedding(v + 1, args.hidden_units, padding_idx=0) for k, v in all_sparse_feats.items() if v > 0})

        self.mm_encoders, self.mm_decoders, self.fusion_transform = [nn.ModuleDict() for _ in range(3)]
        if self.ITEM_EMB_FEAT:
            latent_dim = getattr(args, 'latent_dim', 128)
            hidden_channels = getattr(args, 'mm_hidden_channels', [256])
            for feat_id, input_dim in self.ITEM_EMB_FEAT.items():
                self.mm_encoders[feat_id] = RQEncoder(input_dim, hidden_channels, latent_dim)
                self.mm_decoders[feat_id] = RQDecoder(latent_dim, hidden_channels[::-1], input_dim)
                self.fusion_transform[feat_id] = nn.Linear(latent_dim, args.hidden_units)
            if len(self.ITEM_EMB_FEAT) > 0:
                self.fusion_gate = GatedFusion(len(self.ITEM_EMB_FEAT), args.hidden_units)
        
        user_dim = args.hidden_units * (1 + len(self.USER_SPARSE_FEAT) + len(self.USER_ARRAY_FEAT)) + len(self.USER_CONTINUAL_FEAT)
        item_dim = args.hidden_units * (1 + len(self.ITEM_SPARSE_FEAT) + len(self.ITEM_ARRAY_FEAT)) + len(self.ITEM_CONTINUAL_FEAT)
        self.userdnn = nn.Linear(user_dim, args.hidden_units) if user_dim > args.hidden_units else nn.Identity()
        self.itemdnn = nn.Linear(item_dim, args.hidden_units) if item_dim > args.hidden_units else nn.Identity()

        self.attention_layernorms, self.attention_layers, self.forward_layernorms, self.forward_layers = [nn.ModuleList() for _ in range(4)]
        for _ in range(args.num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.attention_layers.append(FlashMultiHeadAttention(args.hidden_units, args.num_heads, args.dropout_rate))
            self.forward_layernorms.append(nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(args.hidden_units, args.dropout_rate))
        self.last_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)

    def _init_feat_info(self, feat_statistics, feat_types):
        self.USER_SPARSE_FEAT = {k: feat_statistics.get(k, 0) for k in feat_types.get('user_sparse', [])}
        self.USER_CONTINUAL_FEAT = feat_types.get('user_continual', [])
        self.ITEM_SPARSE_FEAT = {k: feat_statistics.get(k, 0) for k in feat_types.get('item_sparse', [])}
        self.ITEM_CONTINUAL_FEAT = feat_types.get('item_continual', [])
        self.USER_ARRAY_FEAT = {k: feat_statistics.get(k, 0) for k in feat_types.get('user_array', [])}
        self.ITEM_ARRAY_FEAT = {k: feat_statistics.get(k, 0) for k in feat_types.get('item_array', [])}
        EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in self.args.mm_emb_id if k in EMB_SHAPE_DICT}
        
    def get_item_embedding(self, item_seq, features: dict):
        base_emb = self.item_emb(item_seq.long())
        
        item_feat_list = [base_emb]
        for feat_id in self.ITEM_SPARSE_FEAT.keys():
            if feat_id in features and feat_id in self.sparse_emb:
                item_feat_list.append(self.sparse_emb[feat_id](features[feat_id].long()))
        for feat_id in self.ITEM_ARRAY_FEAT.keys():
            if feat_id in features and feat_id in self.sparse_emb:
                item_feat_list.append(self.sparse_emb[feat_id](features[feat_id].long()).sum(dim=-2))
        
        base_item_emb = self.itemdnn(torch.cat(item_feat_list, dim=-1))

        semantic_emb, recon_loss = None, torch.tensor(0.0, device=self.dev)
        if self.ITEM_EMB_FEAT:
            raw_feats_dict = {feat_id: features[feat_id].float() for feat_id in self.ITEM_EMB_FEAT if feat_id in features}
            if raw_feats_dict:
                latent_vectors = {feat_id: self.mm_encoders[feat_id](feats) for feat_id, feats in raw_feats_dict.items()}
                recon_loss = sum(F.mse_loss(self.mm_decoders[k](v), raw_feats_dict[k]) for k, v in latent_vectors.items())
                if latent_vectors: recon_loss /= len(latent_vectors)
                
                transformed_latents = [self.fusion_transform[k](v) for k, v in latent_vectors.items()]
                semantic_emb = self.fusion_gate(transformed_latents)

        return base_item_emb + semantic_emb if semantic_emb is not None else base_item_emb, recon_loss

    def log2feats(self, item_seq, features: dict):
        item_embs, recon_loss = self.get_item_embedding(item_seq, features)
        
        seqs = item_embs * (self.item_emb.embedding_dim**0.5)
        batch_size, maxlen = item_seq.shape
        positions = torch.arange(maxlen, device=self.dev).unsqueeze(0).expand(batch_size, -1)
        seqs += self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        timeline_mask = (item_seq != 0)
        attention_mask = torch.tril(torch.ones((maxlen, maxlen), dtype=torch.bool, device=self.dev))
        attention_mask = attention_mask & timeline_mask.unsqueeze(1)
        
        for i in range(len(self.attention_layers)):
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, Q, Q, attn_mask=attention_mask)
            seqs = seqs + mha_outputs
            fwd_input = self.forward_layernorms[i](seqs)
            seqs = seqs + self.forward_layers[i](fwd_input)

        return self.last_layernorm(seqs), recon_loss

    def forward(self, seq, pos, neg, token_type, next_token_type, _, seq_features, pos_features, neg_features):
        user_ids = torch.where(token_type == 2, seq, 0).sum(dim=1)
        
        log_feats, recon_loss_seq = self.log2feats(seq, seq_features)
        print( log_feats.size(), recon_loss_seq.size())
        log_feats += self.user_emb(user_ids.long()).unsqueeze(1)
        
        pos_embs, recon_loss_pos = self.get_item_embedding(pos, pos_features)
        neg_embs, recon_loss_neg = self.get_item_embedding(neg, neg_features)
        print()
        
        total_recon_loss = (recon_loss_seq + recon_loss_pos + recon_loss_neg) / 3.0
        
        return log_feats, pos_embs, neg_embs, total_recon_loss

    def predict(self, log_seqs, seq_features, token_type):
        log_feats, _ = self.log2feats(log_seqs, seq_features)
        
        user_ids = torch.where(token_type == 2, log_seqs, 0).sum(dim=1)
        log_feats += self.user_emb(user_ids.long()).unsqueeze(1)
        
        return log_feats[:, -1, :]
    
    # --- MODIFICATION START ---
    def save_item_emb(self, item_ids, retrieval_ids, features_list, dataset, save_path, batch_size=1024):
        # FIX 1: Update the function signature to match the call in `infer.py`.
        # It now accepts `item_ids`, `retrieval_ids`, and `features_list` directly.
        
        all_embs = []

        self.eval()
        with torch.no_grad():
            # FIX 2: Iterate over the passed arguments, not a self-generated list.
            for i in tqdm(range(0, len(item_ids), batch_size), desc="Saving item embeddings"):
                batch_item_ids = item_ids[i:i+batch_size]
                batch_features_list = features_list[i:i+batch_size]
                
                # FIX 3: Simplified and corrected batch processing.
                # Avoids complex mocking of collate_fn.
                
                # Prepare item ID tensor for the model. Shape: [B, 1]
                batch_ids_tensor = torch.tensor(batch_item_ids, device=self.dev).unsqueeze(1)
                
                # Manually collate features for the batch into the required Dict[str, Tensor] format.
                collated_feats = {}
                # Use all known feature IDs from the dataset to ensure completeness.
                for k in dataset.all_feat_ids:
                    # Gather values for feature 'k' from each item's feature dictionary in the batch.
                    # Use the dataset's default value if a feature is missing.
                    vals = [d.get(k, dataset.feature_default_value.get(k)) for d in batch_features_list]
                    if all(v is None for v in vals): continue
                    
                    try:
                        # Stack numpy arrays (for mm-features) or create a new tensor for other types.
                        if isinstance(vals[0], np.ndarray):
                            tensor = torch.from_numpy(np.stack(vals)).float()
                        else:
                            tensor = torch.tensor(vals, dtype=torch.long)
                        
                        # Reshape to [B, 1, D] (for features) or [B, 1] (for sparse ids) and move to device.
                        # The .unsqueeze(1) adds the sequence length dimension, which is 1 for single item processing.
                        collated_feats[k] = tensor.unsqueeze(1).to(self.dev)
                    except Exception as e:
                        # This helps debug if a specific feature has inconsistent data types.
                        # print(f"Warning: Could not process feature '{k}'. Error: {e}")
                        continue

                # Get embeddings for the batch.
                item_embs, _ = self.get_item_embedding(batch_ids_tensor, collated_feats)
                all_embs.append(item_embs.squeeze(1).detach().cpu().numpy().astype(np.float32))

        final_embs = np.concatenate(all_embs, axis=0)
        final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)

        save_emb(final_embs, Path(save_path) / 'embedding.fbin')
        save_emb(final_ids, Path(save_path) / 'id.u64bin')