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

更新内容以及分数
> 基于上一版本**InfonceLossAddition**代码进行优化，引入clipnorm用于梯度裁剪。，加入tirpletLoss用于优化，设置整体损失计算方法为total_loss = w * loss_infonce + (1 - w) * loss_triplet，其中 w 由 --infonce_loss_weight 参数控制 (默认为0.95)
>>- Score: 0.0510698
>>- NDCG@10: 0.0395379
>>- HitRate@10: 0.0767374

按照时间顺序发布的分支版本依次如下（TripletLoss_And_ClipNorm为最新发布）
  BaselineRMSNorm
  GenerativeFeatureSASRecRQVAE
  InfonceLossAddition
* TripletLoss_And_ClipNorm

版本发布方法：
git checkout -b BranchName
