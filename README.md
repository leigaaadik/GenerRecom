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

> 在上一版本 **InfonceLossAddition** 的基础上进行优化：
> - 引入 `clipnorm` 做梯度裁剪
> - 新增 `TripletLoss` 参与优化
> - 损失函数改为  
>   `total_loss = w * loss_infonce + (1 - w) * loss_triplet`  
>   其中权重 `w` 由参数 `--infonce_loss_weight` 控制（默认 0.95）

| 指标        | 数值      |
|-------------|-----------|
| Score       | 0.0510698 |
| NDCG@10     | 0.0395379 |
| HitRate@10  | 0.0767374 |

---

## 4️⃣ 按时间顺序的分支版本

1. BaselineRMSNorm
2. GenerativeFeatureSASRecRQVAE
3. InfonceLossAddition
4. **TripletLoss_And_ClipNorm**（最新）

---

## 5️⃣ 版本发布方法

```bash
git checkout -b <BranchName>
git add <filename>
git commit -m "<commit message>"
git push origin <BranchName>
