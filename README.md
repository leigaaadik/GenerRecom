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


>- 这一版的模型训练时间极慢，约30min/eposh
>- 实际推理效果不佳，得分较低
>- 拟做消融实验，消除部分特征以及RQVAE组件验证效果
Score: 0.0238746
NDCG@10: 0.018342
HitRate@10: 0.0361892