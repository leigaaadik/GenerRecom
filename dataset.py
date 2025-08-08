import json
import pickle
import struct
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


class MyDataset(torch.utils.data.Dataset):
    """
    基础数据集类，主要为训练服务。
    """
    def __init__(self, data_dir, args):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.maxlen = args.maxlen
        self.mm_emb_id = args.mm_emb_id

        self._load_data_and_offsets()

        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])
            self.usernum = len(indexer['u'])
        self.indexer = indexer
        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}

        self.item_feat_dict = json.load(open(Path(data_dir, "item_feat_dict.json"), 'r'))
        self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_id)

        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()

        self.all_feat_ids = set()
        for f_list in self.feature_types.values():
            self.all_feat_ids.update(f_list)

    def _load_data_and_offsets(self):
        self.data_file_path = self.data_dir / "seq.jsonl"
        self.data_file = None
        with open(self.data_dir / 'seq_offsets.pkl', 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _load_user_data(self, uid):
        if self.data_file is None:
            self.data_file = open(self.data_file_path, 'rb')
        self.data_file.seek(self.seq_offsets[uid])
        line = self.data_file.readline()
        return json.loads(line)

    def _random_neq(self, l, r, s):
        t = np.random.randint(l, r)
        while t in s or str(t) not in self.item_feat_dict:
            t = np.random.randint(l, r)
        return t

    def __getitem__(self, uid):
        user_sequence = self._load_user_data(uid)

        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, _ = record_tuple
            if u and user_feat:
                ext_user_sequence.insert(0, (u, self.fill_missing_feat(user_feat, 0), 2, action_type))
            if i and item_feat:
                ext_user_sequence.append((i, self.fill_missing_feat(item_feat, i), 1, action_type))

        seq, token_type, next_token_type, next_action_type = [], [], [], []
        pos, neg = [], []
        seq_feat, pos_feat, neg_feat = [], [], []

        ts = {rec[0] for rec in ext_user_sequence if rec[2] == 1 and rec[0]}

        for j in range(len(ext_user_sequence) - 1):
            i, feat, type_, act_type = ext_user_sequence[j]
            next_i, next_feat, next_type, next_act_type = ext_user_sequence[j+1]

            seq.append(i)
            token_type.append(type_)
            seq_feat.append(feat)
            next_token_type.append(next_type)
            next_action_type.append(next_act_type or 0)

            if next_type == 1 and next_i != 0:
                pos.append(next_i)
                pos_feat.append(next_feat)
                neg_id = self._random_neq(1, self.itemnum + 1, ts)
                neg.append(neg_id)
                neg_feat.append(self.fill_missing_feat(self.item_feat_dict.get(str(neg_id)), neg_id))
            else:
                pos.append(0)
                pos_feat.append(self.feature_default_value)
                neg.append(0)
                neg_feat.append(self.feature_default_value)

        return (
            seq[-self.maxlen:], token_type[-self.maxlen:], next_token_type[-self.maxlen:], next_action_type[-self.maxlen:],
            pos[-self.maxlen:], neg[-self.maxlen:],
            seq_feat[-self.maxlen:], pos_feat[-self.maxlen:], neg_feat[-self.maxlen:]
        )

    def __len__(self):
        return len(self.seq_offsets)

    def _init_feat_info(self):
        feat_default_value = {}
        feat_statistics = {}
        feat_types = {}
        feat_types['user_sparse'] = ['103', '104', '105', '109']
        feat_types['item_sparse'] = ['100', '117', '111', '118', '101', '102', '119', '120', '114', '112', '121', '115', '122', '116']
        feat_types['item_array'] = []
        feat_types['user_array'] = ['106', '107', '108', '110']
        feat_types['item_emb'] = self.mm_emb_id
        feat_types['user_continual'] = []
        feat_types['item_continual'] = []

        all_known_f_ids = self.indexer['f'].keys()
        for feat_id in feat_types['user_sparse']:
            feat_default_value[feat_id] = 0
            if feat_id in all_known_f_ids: feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_sparse']:
            feat_default_value[feat_id] = 0
            if feat_id in all_known_f_ids: feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_array']:
            feat_default_value[feat_id] = [0]
            if feat_id in all_known_f_ids: feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_array']:
            feat_default_value[feat_id] = [0]
            if feat_id in all_known_f_ids: feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_continual']: feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_continual']: feat_default_value[feat_id] = 0
        if self.mm_emb_dict:
            for feat_id in feat_types['item_emb']:
                if self.mm_emb_dict.get(feat_id):
                    sample_emb = next(iter(self.mm_emb_dict[feat_id].values()), None)
                    if sample_emb is not None:
                         feat_default_value[feat_id] = np.zeros(sample_emb.shape[0], dtype=np.float32)

        return feat_default_value, feat_types, feat_statistics

    def fill_missing_feat(self, feat, item_id):
        if feat is None: feat = {}
        filled_feat = feat.copy()

        missing_fields = self.all_feat_ids - set(feat.keys())
        for feat_id in missing_fields:
            if feat_id in self.feature_default_value:
                filled_feat[feat_id] = self.feature_default_value[feat_id]

        for feat_id in self.feature_types.get('item_emb', []):
            if item_id != 0 and feat_id in self.mm_emb_dict:
                item_original_id = self.indexer_i_rev.get(item_id)
                if item_original_id and item_original_id in self.mm_emb_dict[feat_id]:
                    emb = self.mm_emb_dict[feat_id][item_original_id]
                    if isinstance(emb, np.ndarray):
                        filled_feat[feat_id] = emb
        return filled_feat

    def collate_fn_wrapper(self, batch):
        (seqs, token_types, next_token_types, next_action_types,
         poses, negs,
         seq_feats_list, pos_feats_list, neg_feats_list) = zip(*batch)

        batch_size = len(seqs)
        maxlen = self.maxlen

        padded_seqs = np.zeros((batch_size, maxlen), dtype=np.int64)
        padded_token_types = np.zeros((batch_size, maxlen), dtype=np.int64)
        padded_next_token_types = np.zeros((batch_size, maxlen), dtype=np.int64)
        padded_next_action_types = np.zeros((batch_size, maxlen), dtype=np.int64)
        padded_poses = np.zeros((batch_size, maxlen), dtype=np.int64)
        padded_negs = np.zeros((batch_size, maxlen), dtype=np.int64)

        for i, seq in enumerate(seqs):
            trunc_len = len(seq)
            padded_seqs[i, -trunc_len:] = seq
            padded_token_types[i, -trunc_len:] = token_types[i]
            padded_next_token_types[i, -trunc_len:] = next_token_types[i]
            padded_next_action_types[i, -trunc_len:] = next_action_types[i]
            padded_poses[i, -trunc_len:] = poses[i]
            padded_negs[i, -trunc_len:] = negs[i]

        def process_features(feats_list):
            processed_tensors = {}
            for k in self.all_feat_ids:
                is_array = k in self.feature_types.get('user_array', []) or k in self.feature_types.get('item_array', [])
                is_emb = k in self.feature_types.get('item_emb', [])

                if is_array:
                    max_array_len = max((len(d.get(k, [0])) for seq_f in feats_list for d in seq_f), default=1)
                    tensor = np.zeros((batch_size, maxlen, max_array_len), dtype=np.int64)
                elif is_emb:
                    if k not in self.feature_default_value: continue
                    emb_dim = self.feature_default_value[k].shape[0]
                    tensor = np.zeros((batch_size, maxlen, emb_dim), dtype=np.float32)
                else:
                    tensor = np.zeros((batch_size, maxlen), dtype=np.int64)

                for i, seq_f in enumerate(feats_list):
                    trunc_len = len(seq_f)
                    for j, feat_dict in enumerate(seq_f):
                        val = feat_dict.get(k, self.feature_default_value.get(k))
                        if val is None: continue
                        if is_array:
                            actual_len = min(len(val), max_array_len)
                            tensor[i, maxlen - trunc_len + j, :actual_len] = val[:actual_len]
                        else:
                            tensor[i, maxlen - trunc_len + j] = val
                processed_tensors[k] = torch.from_numpy(tensor)
            return processed_tensors

        seq_features = process_features(seq_feats_list)
        pos_features = process_features(pos_feats_list)
        neg_features = process_features(neg_feats_list)

        return (
            torch.from_numpy(padded_seqs), torch.from_numpy(padded_poses), torch.from_numpy(padded_negs),
            torch.from_numpy(padded_token_types), torch.from_numpy(padded_next_token_types), torch.from_numpy(padded_next_action_types),
            seq_features, pos_features, neg_features
        )


class MyTestDataset(MyDataset):
    """
    测试数据集，继承自MyDataset并重写必要的方法。
    """
    def __init__(self, data_dir, args):
        super().__init__(data_dir, args)

    def _load_data_and_offsets(self):
        self.data_file_path = self.data_dir / "predict_seq.jsonl"
        self.data_file = None
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _process_cold_start_feat(self, feat):
        if feat is None: return {}
        processed_feat = {}
        for feat_id, feat_value in feat.items():
            if isinstance(feat_value, list):
                processed_feat[feat_id] = [0 if isinstance(v, str) else v for v in feat_value]
            elif isinstance(feat_value, str):
                processed_feat[feat_id] = 0
            else:
                processed_feat[feat_id] = feat_value
        return processed_feat

    def __getitem__(self, uid):
        user_sequence = self._load_user_data(uid)
        user_id = ""

        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, _, _ = record_tuple

            if u and isinstance(u, str): user_id = u
            if u and user_id == "" and isinstance(u, int): user_id = self.indexer_u_rev.get(u, "")

            if u and user_feat:
                u_id = 0 if isinstance(u, str) else u
                processed_user_feat = self._process_cold_start_feat(user_feat)
                ext_user_sequence.insert(0, (u_id, self.fill_missing_feat(processed_user_feat, 0), 2))

            if i and item_feat:
                i_id = 0 if i > self.itemnum else i
                processed_item_feat = self._process_cold_start_feat(item_feat)
                ext_user_sequence.append((i_id, self.fill_missing_feat(processed_item_feat, i_id), 1))

        seq = [rec[0] for rec in ext_user_sequence]
        token_type = [rec[2] for rec in ext_user_sequence]
        seq_feat = [rec[1] for rec in ext_user_sequence]

        return (
            seq[-self.maxlen:],
            token_type[-self.maxlen:],
            seq_feat[-self.maxlen:],
            user_id
        )

    def collate_fn_wrapper(self, batch):
        seqs, token_types, seq_feats_list, user_ids = zip(*batch)

        batch_size = len(seqs)
        maxlen = self.maxlen

        padded_seqs = np.zeros((batch_size, maxlen), dtype=np.int64)
        padded_token_types = np.zeros((batch_size, maxlen), dtype=np.int64)

        for i, seq in enumerate(seqs):
            trunc_len = len(seq)
            padded_seqs[i, -trunc_len:] = seq
            padded_token_types[i, -trunc_len:] = token_types[i]

        def process_features(feats_list):
            processed_tensors = {}
            for k in self.all_feat_ids:
                is_array = k in self.feature_types.get('user_array', []) or k in self.feature_types.get('item_array', [])
                is_emb = k in self.feature_types.get('item_emb', [])

                if is_array:
                    max_array_len = max((len(d.get(k, [0])) for seq_f in feats_list for d in seq_f), default=1)
                    tensor = np.zeros((batch_size, maxlen, max_array_len), dtype=np.int64)
                elif is_emb:
                    if k not in self.feature_default_value: continue
                    emb_dim = self.feature_default_value[k].shape[0]
                    tensor = np.zeros((batch_size, maxlen, emb_dim), dtype=np.float32)
                else:
                    tensor = np.zeros((batch_size, maxlen), dtype=np.int64)

                for i, seq_f in enumerate(feats_list):
                    trunc_len = len(seq_f)
                    for j, feat_dict in enumerate(seq_f):
                        val = feat_dict.get(k, self.feature_default_value.get(k))
                        if val is None: continue
                        if is_array:
                            actual_len = min(len(val), max_array_len)
                            tensor[i, maxlen - trunc_len + j, :actual_len] = val[:actual_len]
                        else:
                            tensor[i, maxlen - trunc_len + j] = val
                processed_tensors[k] = torch.from_numpy(tensor)
            return processed_tensors

        seq_features = process_features(seq_feats_list)

        return (
            torch.from_numpy(padded_seqs),
            torch.from_numpy(padded_token_types),
            seq_features,
            list(user_ids)
        )

# --- Helper Functions ---

def save_emb(emb, save_path):
    num_points, num_dimensions = emb.shape[0], emb.shape[1]
    print(f'saving {save_path}')
    with open(Path(save_path), 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions))
        emb.tofile(f)

def load_mm_emb(mm_path, feat_ids):
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    mm_emb_dict = {}
    for feat_id in tqdm(feat_ids, desc='Loading mm_emb'):
        if feat_id not in SHAPE_DICT: continue
        shape = SHAPE_DICT[feat_id]
        emb_dict = {}
        try:
            if feat_id == '81':
                with open(Path(mm_path, f'emb_{feat_id}_{shape}.pkl'), 'rb') as f:
                    emb_dict = pickle.load(f)
            else:
                base_path = Path(mm_path, f'emb_{feat_id}_{shape}')
                if base_path.is_dir():
                    for json_file in base_path.glob('*.json'):
                        with open(json_file, 'r', encoding='utf-8') as file:
                            for line in file:
                                data = json.loads(line.strip())
                                emb = data.get('emb')
                                if isinstance(emb, list): emb = np.array(emb, dtype=np.float32)
                                emb_dict[data.get('anonymous_cid')] = emb
            mm_emb_dict[feat_id] = emb_dict
        except Exception as e:
            print(f"Error loading mm_emb for feat_id {feat_id}: {e}")
    return mm_emb_dict