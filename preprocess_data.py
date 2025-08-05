import json
import pickle
from pathlib import Path
from tqdm import tqdm
import os
import numpy as np
import re

def merge_and_index_sequences(data_dir):
    """
    使用流式解析器，逐个提取和验证JSON对象，以处理任意拼接或无分隔符的JSON数据流。
    """
    print("--- 步骤 1: 开始处理用户序列数据  ---")
    data_path = Path(data_dir)
    
    all_part_files = sorted(list(data_path.glob("**/part-*")))
    seq_part_files = [
        f for f in all_part_files 
        if 'creative_emb' not in str(f) and not f.name.endswith('_SUCCESS')
    ]

    if not seq_part_files:
        print("警告：在目录中找不到用户序列的 'part-*' 文件。")
        return

    print(f"找到了 {len(seq_part_files)} 个用户序列 part 文件，将开始合并和索引...")

    output_seq_file_path = data_path / 'seq.jsonl'
    output_offsets_file_path = data_path / 'seq_offsets.pkl'
    
    offsets = []
    decoder = json.JSONDecoder()
    
    with open(output_seq_file_path, 'wb') as f_out:
        current_offset = 0
        for part_file in tqdm(seq_part_files, desc="合并序列 Part 文件"):
            content = part_file.read_text(encoding='utf-8').strip()
            if not content:
                continue

            pos = 0
            while pos < len(content):
                # 跳过当前位置前的所有空白字符
                match = re.search(r'\S', content[pos:])
                if match is None:
                    break # 字符串剩余部分全是空白
                pos += match.start()

                try:
                    # 使用 raw_decode 从当前位置解析一个JSON对象
                    obj, end_pos = decoder.raw_decode(content, pos)
                    
                    # 将解析出的有效对象重新序列化为标准、紧凑的JSON字符串
                    # 并确保以换行符结尾
                    line_to_write = json.dumps(obj, separators=(',', ':')) + '\n'
                    encoded_line = line_to_write.encode('utf-8')
                    
                    # 记录写入前的偏移量
                    offsets.append(current_offset)
                    # 写入文件
                    f_out.write(encoded_line)
                    # 更新偏移量
                    current_offset += len(encoded_line)
                    
                    # 将解析位置移动到刚解析完的对象的末尾
                    pos = end_pos

                except json.JSONDecodeError:
                    # 如果 raw_decode 失败，说明此处有一个无法解析的片段
                    # 我们跳过这个损坏的片段，寻找下一个可能的JSON对象起始点 '{'
                    next_brace = content.find('{', pos + 1)
                    if next_brace == -1:
                        # 后面再也没有 '{' 了，结束对这个文件的处理
                        break
                    else:
                        print(f"\n警告: 在文件 {part_file} 中跳过一个损坏的数据片段。")
                        pos = next_brace

    print(f"成功合并所有用户序列到: {output_seq_file_path}")
    
    with open(output_offsets_file_path, 'wb') as f_offsets:
        pickle.dump(offsets, f_offsets)
        
    print(f"成功创建序列索引文件: {output_offsets_file_path}")
    print(f"总共处理了 {len(offsets)} 个用户序列。")

def preprocess_embeddings(data_dir):
    """
    处理多模态嵌入数据 
    """
    print("\n--- 步骤 2: 开始处理多模态嵌入数据 ---")
    creative_emb_path = Path(data_dir) / 'creative_emb'
    
    if not creative_emb_path.exists():
        print(f"错误：找不到嵌入目录 {creative_emb_path}。")
        return

    for emb_dir in creative_emb_path.iterdir():
        if emb_dir.is_dir() and emb_dir.name.startswith('emb_'):
            print(f"\n正在处理嵌入类型: {emb_dir.name}")
            part_files = sorted(list(emb_dir.glob("part-*")))
            part_files = [f for f in part_files if not f.name.endswith('_SUCCESS')]
            if not part_files: continue

            embedding_data = {}
            for part_file in tqdm(part_files, desc=f"  -> 读取 {emb_dir.name}"):
                with open(part_file, 'r', encoding='utf-8') as f_in:
                    for line in f_in:
                        try:
                            data_dict = json.loads(line.strip())
                            key_id = next(k for k, v in data_dict.items() if k != 'emb')
                            emb_vector = data_dict['emb']
                            if isinstance(emb_vector, list):
                                emb_vector = np.array(emb_vector, dtype=np.float32)
                            embedding_data[data_dict[key_id]] = emb_vector
                        except Exception:
                            pass # 静默处理嵌入文件中的个别错误行
            
            output_pkl_path = creative_emb_path / f"{emb_dir.name}.pkl"
            with open(output_pkl_path, 'wb') as f_out:
                pickle.dump(embedding_data, f_out)
            print(f"  -> 成功创建嵌入文件: {output_pkl_path}，包含 {len(embedding_data)} 个条目。")

if __name__ == '__main__':
    DATA_DIRECTORY = 'TencentGR_1k'
    
    if not Path(DATA_DIRECTORY).exists():
        print(f"错误：数据目录 '{DATA_DIRECTORY}' 不存在。请检查路径配置。")
    else:
        merge_and_index_sequences(DATA_DIRECTORY)
        preprocess_embeddings(DATA_DIRECTORY)
        
        print("\n\n✅  所有预处理已完成！现在项目可以正常运行。")