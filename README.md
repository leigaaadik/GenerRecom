# 🚀 项目启动指南

## 1️⃣ 数据预处理
运行下列命令，将原始分片数据整理成模型可用的格式：

```bash
python preprocess_data.py
```

完成后，目录结构如下（`tree -L 2`）：

```
.
├── creative_emb
│   ├── emb_81_32
│   ├── emb_81_32.pkl
│   ├── emb_82_1024
│   ├── emb_82_1024.pkl
│   ├── emb_83_3584
│   ├── emb_83_3584.pkl
│   ├── emb_84_4096
│   ├── emb_84_4096.pkl
│   ├── emb_85_4096
│   ├── emb_85_4096.pkl
│   ├── emb_86_3584
│   └── emb_86_3584.pkl
├── indexer.pkl
├── item_feat_dict.json
├── README.md
├── seq.jsonl
└── seq_offsets.pkl

7 directories, 11 files
```

## 2️⃣ 训练启动
预处理完成后，启动训练脚本：

```bash
chmod +x run.sh     # 修改执行权限
bash run.sh         # 开始训练
```

脚本会自动设置所有环境变量并调用 `main.py`；  

---

## 3️⃣ 更新内容与性能指标

> 在上一版本 **TripletLoss_And_ClipNorm** 的基础上进行优化：
> - 删除TripletLoss，仅保留InfonceLoss作为损失函数
> - 修改模型架构，序列架构替换为HSTU
> - 修改原HSTU中的绝对位置编码改为利用timestamp进行RoPE旋转位置编码，修改归一化方法LayerNorm->RMSNorm
> - 学习率改为改为预热warm up+线性衰减weight deacy
> - 修改dataset.py从数据中读出timestamp特征
> - 修改代码框架，可视化正负样本相似度以及区分度

> 表格中记录了训练产出的第3个epoch和第4个epoch模型的评测结果，说明模型仍在收敛，可以通过增加训练epoch数提升得分

> 训练时长约1h30min/epoch

| 指标(epoch4)| 数值      |
|-------------|-----------|
| Score       | 0.0555635 |
| NDCG@10     | 0.0431958 |
| HitRate@10  | 0.0830917 |

| 指标(epoch3)| 数值      |
|-------------|-----------|
| Score       | 0.0537773 |
| NDCG@10     | 0.0418228 |
| HitRate@10  | 0.0803857 |

---

## 4️⃣ 按时间顺序的分支版本


1. GenerativeFeatureSASRecRQVAE
2. BaselineRMSNorm
3. InfonceLossAddition
4. TripletLoss_And_ClipNorm
5. **HSTU**（最新）

---

## 5️⃣ 版本发布方法

```bash
git checkout -b <BranchName>
git add <filename>
git commit -m "<commit message>"
git push origin <BranchName>
