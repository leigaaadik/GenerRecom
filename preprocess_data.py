import json
import pickle
from pathlib import Path
from tqdm import tqdm
import os
import numpy as np

def merge_and_index_sequences(data_dir):
    """
    读取指定目录下的所有与序列相关的 part-* 文件，将它们合并成一个 seq.jsonl 文件，
    并为每一行生成字节偏移量索引，保存为 seq_offsets.pkl。
    """
    print("--- 步骤 1: 开始处理用户序列数据 ---")
    data_path = Path(data_dir)
    
    # 根据您的文件结构，part文件似乎分散在多个子目录中。
    # 我们需要找到包含用户序列的part文件。假设它们直接在主数据目录下或特定子目录。
    # 为了避免混淆嵌入文件的part文件，我们假设序列文件不包含 'emb' 在路径中。
    # !! 如果您的序列part文件在特定文件夹，请修改这里的glob模式 !!
    # 例如：part_files = sorted(list(data_path.glob("raw_sequences/part-*")))
    all_part_files = sorted(list(data_path.glob("**/part-*")))
    
    # 过滤掉属于creative_emb的part文件和_SUCCESS文件
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
    遍历 creative_emb 目录，将每个 emb_XX_YY/part-* 目录下的JSON行
    合并成一个单一的 emb_XX_YY.pkl 文件。
    """
    print("\n--- 步骤 2: 开始处理多模态嵌入数据 ---")
    creative_emb_path = Path(data_dir) / 'creative_emb'
    
    if not creative_emb_path.exists():
        print(f"错误：找不到嵌入目录 {creative_emb_path}。")
        return

    # 遍历 creative_emb 下的所有子目录，例如 emb_81_32, emb_82_1024 ...
    for emb_dir in creative_emb_path.iterdir():
        if emb_dir.is_dir() and emb_dir.name.startswith('emb_'):
            print(f"\n正在处理嵌入类型: {emb_dir.name}")
            
            part_files = sorted(list(emb_dir.glob("part-*")))
            part_files = [f for f in part_files if not f.name.endswith('_SUCCESS')]

            if not part_files:
                print(f"  -> 在 {emb_dir} 中找不到 part 文件，跳过。")
                continue

            # 这是我们将要构建的字典，key是物品ID，value是嵌入向量
            embedding_data = {}
            
            for part_file in tqdm(part_files, desc=f"  -> 读取 {emb_dir.name}"):
                with open(part_file, 'r', encoding='utf-8') as f_in:
                    for line in f_in:
                        try:
                            data_dict = json.loads(line.strip())
                            # 键可能是 'anonymous_cid' 或其他，我们取第一个非'emb'的键
                            key_id = next(k for k, v in data_dict.items() if k != 'emb')
                            
                            emb_vector = data_dict['emb']
                            
                            # 转换为Numpy数组以获得更好的性能和兼容性
                            if isinstance(emb_vector, list):
                                emb_vector = np.array(emb_vector, dtype=np.float32)
                            
                            embedding_data[data_dict[key_id]] = emb_vector
                        except (json.JSONDecodeError, StopIteration, KeyError) as e:
                            print(f"    警告：在文件 {part_file} 中解析行失败: {line.strip()}. 错误: {e}")
            
            # 定义输出的 .pkl 文件路径
            # 输出文件将位于 creative_emb 目录下，名为 emb_81_32.pkl
            output_pkl_path = creative_emb_path / f"{emb_dir.name}.pkl"

            with open(output_pkl_path, 'wb') as f_out:
                pickle.dump(embedding_data, f_out)
            
            print(f"  -> 成功创建嵌入文件: {output_pkl_path}，包含 {len(embedding_data)} 个条目。")

if __name__ == '__main__':
    # --- 请在这里配置您的数据目录 ---
    # 这个路径应该指向包含 creative_emb, indexer.pkl 等文件的目录
    DATA_DIRECTORY = 'TencentGR_1k'
    
    if not Path(DATA_DIRECTORY).exists():
        print(f"错误：数据目录 '{DATA_DIRECTORY}' 不存在。请检查路径配置。")
    else:
        # 依次执行两个预处理步骤
        merge_and_index_sequences(DATA_DIRECTORY)
        preprocess_embeddings(DATA_DIRECTORY)
        
        print("\n\n✅  所有预处理已完成！现在项目可以正常运行。")