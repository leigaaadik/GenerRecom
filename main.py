# file: main.py

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MyDataset
# 假设您的模型文件名是 model.py
from model import GenerativeFeatureSASRec

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)
    parser.add_argument('--num_epochs', default=20, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    
    # 注意: Baseline的log频率是按step, 这里保留您的设计
    parser.add_argument('--log_freq', default=100, type=int, help='Log training status every N steps.')

    parser.add_argument('--hidden_units', default=128, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--norm_first', action='store_true')

    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])
    parser.add_argument('--latent_dim', default=128, type=int, help='Latent dimension for multi-modal autoencoders')
    parser.add_argument('--mm_hidden_channels', nargs='+', type=int, default=[256], help='Hidden channels for autoencoders')
    parser.add_argument('--triplet_margin', default=1.0, type=float, help='Margin for TripletMarginLoss')
    parser.add_argument('--recon_loss_weight', default=0.1, type=float, help='Weight for reconstruction loss')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    
    # --- MODIFICATION: 路径管理严格遵守Baseline规范 ---
    log_dir = Path(os.environ.get('TRAIN_LOG_PATH'))
    tf_events_dir = Path(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    ckpt_dir = Path(os.environ.get('TRAIN_CKPT_PATH'))
    data_path = os.environ.get('TRAIN_DATA_PATH')

    log_dir.mkdir(parents=True, exist_ok=True)
    tf_events_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True) # 确保根目录存在

    log_file = open(log_dir / 'train.log', 'w')
    writer = SummaryWriter(str(tf_events_dir))

    dataset = MyDataset(data_path, args)
    
    torch.manual_seed(42)
    # 拆分数据集
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    
    # --- MODIFICATION: 更新DataLoader以适配新的collate_fn ---
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        collate_fn=train_dataset.dataset.collate_fn_wrapper, 
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=valid_dataset.dataset.collate_fn_wrapper, 
        pin_memory=True
    )
    
    model = GenerativeFeatureSASRec(dataset.usernum, dataset.itemnum, dataset.feat_statistics, dataset.feature_types, args).to(args.device)
    
    # 模型初始化
    for name, param in model.named_parameters():
        try: torch.nn.init.xavier_normal_(param.data)
        except: pass
    if hasattr(model, 'pos_emb'): model.pos_emb.weight.data[0, :] = 0
    if hasattr(model, 'item_emb'): model.item_emb.weight.data[0, :] = 0
    if hasattr(model, 'user_emb'): model.user_emb.weight.data[0, :] = 0
    if hasattr(model, 'sparse_emb'):
        for k in model.sparse_emb: model.sparse_emb[k].weight.data[0, :] = 0

    epoch_start_idx = 1
    if args.state_dict_path:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=args.device))
            # 尝试从文件名解析epoch，如果格式不同则从1开始
            tail = args.state_dict_path.split('epoch=')[1]
            epoch_start_idx = int(tail.split('.')[0]) + 1
        except Exception:
            print(f'Failed to parse epoch from {args.state_dict_path}, starting from epoch 1.')

    triplet_criterion = torch.nn.TripletMarginLoss(margin=args.triplet_margin, reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    global_step = 0
    
    print("Start training with GenerativeFeatureSASRec model.\n")
    
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        if args.inference_only: break
        
        total_loss_sum, rec_loss_sum, recon_loss_sum, train_step_count = 0.0, 0.0, 0.0, 0
        
        tqdm_loader = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs} [Train]")
        for step, batch in enumerate(tqdm_loader):
            seq, pos, neg, token_type, next_token_type, _, seq_feat, pos_feat, neg_feat = batch
            
            # 将数据移动到设备
            seq, pos, neg, token_type, next_token_type = [x.to(args.device) for x in (seq, pos, neg, token_type, next_token_type)]
            seq_feat = {k: v.to(args.device) for k, v in seq_feat.items()}
            pos_feat = {k: v.to(args.device) for k, v in pos_feat.items()}
            neg_feat = {k: v.to(args.device) for k, v in neg_feat.items()}
            
            anchor_embs, pos_embs, neg_embs, recon_loss = model(
                seq, pos, neg, token_type, next_token_type, None, seq_feat, pos_feat, neg_feat
            )
            
            optimizer.zero_grad()
            indices = torch.where(next_token_type.flatten() == 1)[0]
            
            loss_val, rec_loss_val, recon_loss_val = 0.0, 0.0, 0.0
            if len(indices) > 0:
                rec_loss = triplet_criterion(
                    anchor_embs.view(-1, anchor_embs.shape[-1])[indices], 
                    pos_embs.view(-1, pos_embs.shape[-1])[indices], 
                    neg_embs.view(-1, neg_embs.shape[-1])[indices]
                )
                loss = rec_loss + args.recon_loss_weight * recon_loss
                
                if args.l2_emb > 0:
                    # 假设item_emb是主要的嵌入层
                    for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
                
                loss.backward()
                optimizer.step()

                loss_val, rec_loss_val, recon_loss_val = loss.item(), rec_loss.item(), recon_loss.item()
                total_loss_sum += loss_val
                rec_loss_sum += rec_loss_val
                recon_loss_sum += recon_loss.item()
                train_step_count += 1
                
                # --- MODIFICATION: 按Baseline格式写入日志文件 ---
                if (step + 1) % args.log_freq == 0:
                    log_json = json.dumps({
                        'global_step': global_step, 
                        'loss': loss_val, 
                        'rec_loss': rec_loss_val, 
                        'recon_loss': recon_loss_val, 
                        'epoch': epoch, 
                        'time': time.time()
                    })
                    log_file.write(log_json + '\n')
                    log_file.flush()
                    print(log_json) # 同时打印到控制台，方便调试

            writer.add_scalar('Loss/train_step_total', loss_val, global_step)
            tqdm_loader.set_postfix(loss=loss_val)
            global_step += 1
        
        avg_train_total_loss = total_loss_sum / train_step_count if train_step_count > 0 else 0
        avg_train_rec_loss = rec_loss_sum / train_step_count if train_step_count > 0 else 0
        avg_train_recon_loss = recon_loss_sum / train_step_count if train_step_count > 0 else 0
        writer.add_scalar('Loss/train_epoch_total', avg_train_total_loss, epoch)
        
        # Validation loop
        model.eval()
        valid_loss_sum, valid_rec_loss_sum, valid_recon_loss_sum, valid_step_count = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"Epoch {epoch}/{args.num_epochs} [Valid]"):
                seq, pos, neg, token_type, next_token_type, _, seq_feat, pos_feat, neg_feat = batch
                seq, pos, neg, token_type, next_token_type = [x.to(args.device) for x in (seq, pos, neg, token_type, next_token_type)]
                seq_feat = {k: v.to(args.device) for k, v in seq_feat.items()}
                pos_feat = {k: v.to(args.device) for k, v in pos_feat.items()}
                neg_feat = {k: v.to(args.device) for k, v in neg_feat.items()}

                anchor_embs, pos_embs, neg_embs, recon_loss = model(
                    seq, pos, neg, token_type, next_token_type, None, seq_feat, pos_feat, neg_feat
                )
                
                indices = torch.where(next_token_type.flatten() == 1)[0]
                if len(indices) > 0:
                    rec_loss = triplet_criterion(
                        anchor_embs.view(-1, anchor_embs.shape[-1])[indices], 
                        pos_embs.view(-1, pos_embs.shape[-1])[indices], 
                        neg_embs.view(-1, neg_embs.shape[-1])[indices]
                    )
                    valid_loss_sum += (rec_loss + args.recon_loss_weight * recon_loss).item()
                    valid_rec_loss_sum += rec_loss.item()
                    valid_recon_loss_sum += recon_loss.item()
                    valid_step_count +=1
        
        avg_valid_total_loss = valid_loss_sum / valid_step_count if valid_step_count > 0 else 0
        writer.add_scalar('Loss/valid_epoch_total', avg_valid_total_loss, epoch)

        tqdm.write("-" * 100)
        tqdm.write(f"Epoch: {epoch:03d} | "
                   f"Train Loss: {avg_train_total_loss:.4f} | "
                   f"Valid Loss: {avg_valid_total_loss:.4f}")
        
        # --- MODIFICATION: 模型保存严格遵守Baseline规范 ---
        save_dir = Path(ckpt_dir, f"global_step{global_step}.valid_loss={avg_valid_total_loss:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")
        tqdm.write(f"Model saved to: {save_dir}")
        tqdm.write("-" * 100)
        
    tqdm.write("\n" + "="*40)
    tqdm.write("Training Finished!")
    tqdm.write("="*40)

    writer.close()
    log_file.close()