import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MyTestDataset, save_emb
from model import GenerativeFeatureSASRec


def get_ckpt_path():
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if ckpt_path is None:
        raise ValueError("MODEL_OUTPUT_PATH environment variable is not set")
    for item in os.listdir(ckpt_path):
        if item.endswith(".pt"):
            return os.path.join(ckpt_path, item)
    raise FileNotFoundError(f"No .pt file found in {ckpt_path}")


def get_args():
    """获取与训练时兼容的参数"""
    parser = argparse.ArgumentParser()
    # 基本参数
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--maxlen', default=101, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    
    # 模型结构参数 (与main.py同步)
    parser.add_argument('--hidden_units', default=128, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--norm_first', action='store_true')

    # 多模态与损失函数参数 (与main.py同步)
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])
    parser.add_argument('--latent_dim', default=128, type=int)
    parser.add_argument('--mm_hidden_channels', nargs='+', type=int, default=[256])

    # 未在推理中直接使用但为了兼容性保留的参数
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--num_epochs', default=20, type=int)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--inference_only', action='store_true', default=True)
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--triplet_margin', default=1.0, type=float)
    parser.add_argument('--recon_loss_weight', default=0.1, type=float)

    return parser.parse_args()


def read_result_ids(file_path):
    """读取FAISS生成的二进制结果文件"""
    with open(file_path, 'rb') as f:
        num_points_query = struct.unpack('I', f.read(4))[0]
        query_ann_top_k = struct.unpack('I', f.read(4))[0]
        print(f"Reading ANN results: num_queries={num_points_query}, top_k={query_ann_top_k}")
        num_result_ids = num_points_query * query_ann_top_k
        result_ids = np.fromfile(f, dtype=np.uint64, count=num_result_ids)
        return result_ids.reshape((num_points_query, query_ann_top_k))


def get_candidate_emb(dataset, model):
    """
    生产候选库item的id和embedding

    Args:
        indexer: 索引字典
        feat_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feature_default_value: 特征缺省值
        mm_emb_dict: 多模态特征字典
        model: 模型
    Returns:
        retrieve_id2creative_id: 索引id->creative_id的dict
    """
    eval_data_path = Path(os.environ.get('EVAL_DATA_PATH'))
    eval_result_path = Path(os.environ.get('EVAL_RESULT_PATH'))
    eval_result_path.mkdir(parents=True, exist_ok=True)
    
    candidate_path = eval_data_path / 'predict_set.jsonl'
    
    item_ids, retrieval_ids, features_list = [], [], []
    retrieve_id2creative_id = {}

    with open(candidate_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            creative_id = data['creative_id']
            retrieval_id = data['retrieval_id']
            
            item_id = dataset.indexer['i'].get(creative_id, 0)
            features = dataset._process_cold_start_feat(data.get('features', {}))
            filled_features = dataset.fill_missing_feat(features, item_id)
            
            item_ids.append(item_id)
            retrieval_ids.append(retrieval_id)
            features_list.append(filled_features)
            retrieve_id2creative_id[retrieval_id] = creative_id

    if hasattr(model, 'save_item_emb'):
        model.save_item_emb(item_ids, retrieval_ids, features_list, dataset, str(eval_result_path))
    else:
        print("Warning: model.save_item_emb not found. Using generic predict method.")
        all_embs = []
        pass

    # 保存ID映射文件
    with open(eval_result_path / "retrive_id2creative_id.json", "w") as f:
        json.dump(retrieve_id2creative_id, f)
        
    return retrieve_id2creative_id


def infer():
    """主推理函数"""
    args = get_args()
    
    data_path = Path(os.environ.get('EVAL_DATA_PATH'))
    result_path = Path(os.environ.get('EVAL_RESULT_PATH'))
    result_path.mkdir(parents=True, exist_ok=True)
    
    test_dataset = MyTestDataset(data_path, args)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=test_dataset.collate_fn_wrapper,
        pin_memory=True
    )
    
    model = GenerativeFeatureSASRec(
        test_dataset.usernum, 
        test_dataset.itemnum, 
        test_dataset.feat_statistics, 
        test_dataset.feature_types, 
        args
    ).to(args.device)
    model.eval()

    ckpt_path = get_ckpt_path()
    print(f"Loading model checkpoint from: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(args.device)))
    
    all_user_embs = []
    all_user_ids = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating User Embeddings"):
            seq, token_type, seq_feat, user_ids_batch = batch
            
            seq, token_type = seq.to(args.device), token_type.to(args.device)
            seq_feat = {k: v.to(args.device) for k, v in seq_feat.items()}

            user_embs = model.predict(seq, seq_feat, token_type)
            
            all_user_embs.append(user_embs.cpu().numpy())
            all_user_ids.extend(user_ids_batch)

    all_user_embs = np.concatenate(all_user_embs, axis=0).astype(np.float32)
    save_emb(all_user_embs, result_path / 'query.fbin')

    print("Generating candidate item embeddings...")
    retrieve_id2creative_id = get_candidate_emb(test_dataset, model)

    print("Performing ANN search with FAISS...")
    dataset_vector_file = result_path / "embedding.fbin"
    dataset_id_file = result_path / "id.u64bin"
    query_vector_file = result_path / "query.fbin"
    result_id_file = result_path / "id100.u64bin"

    ann_cmd = (
        f"/workspace/faiss-based-ann/faiss_demo "
        f"--dataset_vector_file_path={dataset_vector_file} "
        f"--dataset_id_file_path={dataset_id_file} "
        f"--query_vector_file_path={query_vector_file} "
        f"--result_id_file_path={result_id_file} "
        f"--query_ann_top_k=10 --faiss_M=64 --faiss_ef_construction=1280 "
        f"--query_ef_search=640 --faiss_metric_type=0"
    )
    os.system(ann_cmd)

    print("Processing search results...")
    top10s_retrieved = read_result_ids(result_id_file)
    final_results = []
    for top10_retrieval_ids in tqdm(top10s_retrieved, desc="Mapping result IDs"):
        user_top10 = [retrieve_id2creative_id.get(int(rid), 0) for rid in top10_retrieval_ids]
        final_results.append(user_top10)

    return final_results, all_user_ids

if __name__ == "__main__":
    top10s, user_list = infer()
    print(f"\nInference finished. Generated recommendations for {len(user_list)} users.")
    for i in range(min(5, len(user_list))):
        print(f"User: {user_list[i]}, Recommendations: {top10s[i]}")