import json
import pickle
from pathlib import Path
from tqdm import tqdm
import os
import numpy as np
import re
import shutil

def merge_and_index_sequences(data_dir):
    """
    读取指定目录下的所有与序列相关的 part-* 文件，将它们合并成一个 seq.jsonl 文件，
    并为每一行生成字节偏移量索引，保存为 seq_offsets.pkl。
    """
    print("--- 步骤 1: 开始处理用户序列数据 ---")
    data_path = Path(data_dir)
    
    all_part_files = sorted(list(data_path.glob("**/part-*")))
    
    seq_part_files = [
        f for f in all_part_files 
        if 'creative_emb' not in str(f) and not f.name.endswith('_SUCCESS')
    ]

    if not seq_part_files:
        print(f"警告：在目录 {data_path} 中找不到用户序列的 'part-*' 文件。")
        print("如果您已手动创建了 seq.jsonl, 可以忽略此警告。")
        return

    print(f"找到了 {len(seq_part_files)} 个用户序列 part 文件，将开始合并和索引...")

    output_seq_file_path = data_path / 'seq.jsonl'
    output_offsets_file_path = data_path / 'seq_offsets.pkl'
    
    offsets = []
    
    with open(output_seq_file_path, 'wb') as f_out:
        current_offset = 0
        for part_file in tqdm(seq_part_files, desc="合并序列 Part 文件"):
            with open(part_file, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    offsets.append(current_offset)
                    encoded_line = line.encode('utf-8')
                    f_out.write(encoded_line)
                    current_offset += len(encoded_line)

    print(f"成功合并所有用户序列到: {output_seq_file_path}")
    
    with open(output_offsets_file_path, 'wb') as f_offsets:
        pickle.dump(offsets, f_offsets)
        
    print(f"成功创建序列索引文件: {output_offsets_file_path}")
    print(f"总共处理了 {len(offsets)} 个用户序列。")

def preprocess_embeddings(data_dir):
    """
    遍历 creative_emb 目录，根据 dataset.py 的加载逻辑生成目标文件。
    - 对于 emb_81_32，合并 part-* 文件到一个 .pkl 文件。
    - 对于其他 emb_XX_YY 目录，合并 part-* 文件到一个 .json 文件。
    """
    print("\n--- 步骤 2: 开始处理多模态嵌入数据 ---")
    creative_emb_path = Path(data_dir) / 'creative_emb'
    
    if not creative_emb_path.exists():
        print(f"错误：找不到嵌入目录 {creative_emb_path}。")
        return

    pattern = re.compile(r'emb_(\d+)_.*')

    for emb_dir in creative_emb_path.iterdir():
        if not emb_dir.is_dir() or not emb_dir.name.startswith('emb_'):
            continue

        match = pattern.match(emb_dir.name)
        if not match:
            print(f"警告：无法从目录名 {emb_dir.name} 中解析特征ID，跳过。")
            continue
            
        feat_id = match.group(1)
        print(f"\n正在处理嵌入类型: {emb_dir.name} (ID: {feat_id})")
        
        part_files = sorted(list(emb_dir.glob("part-*")))
        part_files = [f for f in part_files if not f.name.endswith('_SUCCESS')]

        if not part_files:
            print(f"  -> 在 {emb_dir} 中找不到 part 文件，跳过。")
            continue

        # ==================== MODIFICATION START ====================
        if feat_id == '81':
            # --- 逻辑 1: 处理81号特征，合并为 .pkl ---
            embedding_data = {}
            for part_file in tqdm(part_files, desc=f"  -> 读取并合并为 .pkl"):
                with open(part_file, 'r', encoding='utf-8') as f_in:
                    for line in f_in:
                        try:
                            data_dict = json.loads(line.strip())
                            key_id = data_dict['anonymous_cid']
                            emb_vector = data_dict['emb']
                            if isinstance(emb_vector, list):
                                emb_vector = np.array(emb_vector, dtype=np.float32)
                            embedding_data[key_id] = emb_vector
                        except (json.JSONDecodeError, KeyError) as e:
                            print(f"    警告：在文件 {part_file} 中解析行失败: {line.strip()}. 错误: {e}")
            
            output_pkl_path = creative_emb_path / f"{emb_dir.name}.pkl"
            with open(output_pkl_path, 'wb') as f_out:
                pickle.dump(embedding_data, f_out)
            
            print(f"  -> 成功创建 .pkl 文件: {output_pkl_path}，包含 {len(embedding_data)} 个条目。")
        
        else: # (feat_id is '82', '83', etc.)
            # --- 逻辑 2: 处理其他特征，合并为单一的 .json 文件 ---
            # dataset.py期望在子目录中找到.json文件，我们将所有part合并成一个，简化结构
            output_json_path = emb_dir / 'part-00000.json'
            
            # 备份原始文件
            backup_dir = emb_dir.parent / f"{emb_dir.name}_backup"
            if not backup_dir.exists():
                shutil.move(emb_dir, backup_dir)
                emb_dir.mkdir()
                print(f"  -> 已将原始part文件备份到: {backup_dir}")

            print(f"  -> 开始合并part文件到: {output_json_path}")
            with open(output_json_path, 'w', encoding='utf-8') as f_out:
                for part_file in tqdm(backup_dir.glob("part-*"), desc=f"  -> 合并为 .json"):
                    if part_file.name.endswith('_SUCCESS'): continue
                    with open(part_file, 'r', encoding='utf-8') as f_in:
                        for line in f_in:
                            f_out.write(line) # 直接写入原始的JSON行
            
            print(f"  -> 成功创建 .json 文件: {output_json_path}")

        # ===================== MODIFICATION END =====================

if __name__ == '__main__':
    DATA_DIRECTORY = 'TencentGR_1k'
    
    if not Path(DATA_DIRECTORY).exists():
        print(f"错误：数据目录 '{DATA_DIRECTORY}' 不存在。请检查路径配置。")
    else:
        merge_and_index_sequences(DATA_DIRECTORY)
        preprocess_embeddings(DATA_DIRECTORY)
        
        print("\n\n✅  所有预处理已完成！数据格式现在与代码框架完全匹配。")