# file: main.py 

import argparse
import json
import os
import time
from pathlib import Path
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MyDataset
from model import GenerativeFeatureSASRec


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)
    parser.add_argument('--hidden_units', default=128, type=int)
    parser.add_argument('--num_blocks', default=4, type=int)
    parser.add_argument('--num_epochs', default=50, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float) # 保留此参数以防万一
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])
    parser.add_argument('--latent_dim', default=128, type=int)
    parser.add_argument('--mm_hidden_channels', nargs='+', type=int, default=[256])
    parser.add_argument('--triplet_margin', default=1.0, type=float, help='Margin for TripletMarginLoss')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    log_path = Path(os.environ.get('TRAIN_LOG_PATH', f'./logs/{args.hidden_units}_{args.num_blocks}'))
    tf_events_path = Path(os.environ.get('TRAIN_TF_EVENTS_PATH', f'./tf_events/{args.hidden_units}_{args.num_blocks}'))
    ckpt_path = Path(os.environ.get('TRAIN_CKPT_PATH', f'./ckpts/{args.hidden_units}_{args.num_blocks}'))
    log_path.mkdir(parents=True, exist_ok=True); tf_events_path.mkdir(parents=True, exist_ok=True); ckpt_path.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path / 'train.log', 'w')
    writer = SummaryWriter(str(tf_events_path))
    data_path = os.environ.get('TRAIN_DATA_PATH', './data')

    dataset = MyDataset(data_path, args)
    torch.manual_seed(42)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    
    # 建议在平台上将worker数量固定为一个较小的值
    data_loader_workers = min(4, os.cpu_count() or 1)
    print(f"Using {data_loader_workers} workers for data loading.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=data_loader_workers, collate_fn=dataset.collate_fn_optimized, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=data_loader_workers, collate_fn=dataset.collate_fn_optimized, pin_memory=True)
    
    model = GenerativeFeatureSASRec(dataset.usernum, dataset.itemnum, dataset.feat_statistics, dataset.feature_types, args).to(args.device)
    
    # ... (模型初始化等代码保持不变) ...
    for name, param in model.named_parameters():
        try: torch.nn.init.xavier_normal_(param.data)
        except: pass
    model.pos_emb.weight.data[0, :] = 0; model.item_emb.weight.data[0, :] = 0; model.user_emb.weight.data[0, :] = 0
    for k in model.sparse_emb: model.sparse_emb[k].weight.data[0, :] = 0
    epoch_start_idx = 1
    if args.state_dict_path:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=args.device))
            epoch_start_idx = int(args.state_dict_path.split('epoch=')[1].split('.')[0]) + 1
        except: print(f'Failed loading state_dicts from: {args.state_dict_path}')

    triplet_criterion = torch.nn.TripletMarginLoss(margin=args.triplet_margin, reduction='mean')
    print(f"Using TripletMarginLoss with margin={args.triplet_margin}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    best_valid_loss, best_epoch, global_step = float('inf'), 0, 0
    last_best_ckpt_dir = None # 用于追踪上一个最佳模型的目录
    
    print("Start training\n")
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        if args.inference_only: break
        
        total_loss_sum, rec_loss_sum, recon_loss_sum, train_step_count = 0.0, 0.0, 0.0, 0
        tqdm_loader = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs} [Train]")
        
        for batch in tqdm_loader:
            seq, pos, neg, token_type, next_token_type, _, seq_feat, pos_feat, neg_feat = batch
            seq, pos, neg, next_token_type = seq.to(args.device), pos.to(args.device), neg.to(args.device), next_token_type.to(args.device)
            anchor_embs, pos_embs, neg_embs, recon_loss = model(seq, pos, neg, token_type, next_token_type, None, seq_feat, pos_feat, neg_feat)
            optimizer.zero_grad()
            indices = torch.where(next_token_type == 1)
            rec_loss = triplet_criterion(anchor_embs[indices], pos_embs[indices], neg_embs[indices])
            loss = rec_loss + recon_loss
            if args.l2_emb > 0:
                for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward(); optimizer.step()
            total_loss_sum += loss.item()
            rec_loss_sum += rec_loss.item()
            recon_loss_sum += recon_loss.item()
            train_step_count += 1
            writer.add_scalar('Loss/train_step', loss.item(), global_step)
            global_step += 1
            tqdm_loader.set_postfix(loss=loss.item())

        avg_total_loss = total_loss_sum / train_step_count if train_step_count > 0 else 0
        avg_rec_loss = rec_loss_sum / train_step_count if train_step_count > 0 else 0
        avg_recon_loss = recon_loss_sum / train_step_count if train_step_count > 0 else 0
        writer.add_scalar('Loss/train_epoch_total', avg_total_loss, epoch)
        writer.add_scalar('Loss/train_epoch_rec', avg_rec_loss, epoch)
        writer.add_scalar('Loss/train_epoch_recon', avg_recon_loss, epoch)

        model.eval()
        valid_loss_sum, valid_rec_loss_sum, valid_recon_loss_sum = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"Epoch {epoch}/{args.num_epochs} [Valid]"):
                seq, pos, neg, token_type, next_token_type, _, seq_feat, pos_feat, neg_feat = batch
                seq, pos, neg, next_token_type = seq.to(args.device), pos.to(args.device), neg.to(args.device), next_token_type.to(args.device)
                anchor_embs, pos_embs, neg_embs, recon_loss = model(seq, pos, neg, token_type, next_token_type, None, seq_feat, pos_feat, neg_feat)
                indices = torch.where(next_token_type == 1)
                rec_loss = triplet_criterion(anchor_embs[indices], pos_embs[indices], neg_embs[indices])
                valid_loss_sum += (rec_loss + recon_loss).item()
                valid_rec_loss_sum += rec_loss.item()
                valid_recon_loss_sum += recon_loss.item()
        num_valid_batches = len(valid_loader) if valid_loader else 1
        valid_loss_avg = valid_loss_sum / num_valid_batches
        valid_rec_loss_avg = valid_rec_loss_sum / num_valid_batches
        valid_recon_loss_avg = valid_recon_loss_sum / num_valid_batches
        writer.add_scalar('Loss/valid_epoch_total', valid_loss_avg, epoch)
        writer.add_scalar('Loss/valid_epoch_rec', valid_rec_loss_avg, epoch)
        writer.add_scalar('Loss/valid_epoch_recon', valid_recon_loss_avg, epoch)
        print(f"Epoch: {epoch:03d} | Train Loss: {avg_total_loss:.4f} (Rec: {avg_rec_loss:.4f}, Recon: {avg_recon_loss:.4f}) "
              f"| Valid Loss: {valid_loss_avg:.4f} (Rec: {valid_rec_loss_avg:.4f}, Recon: {valid_recon_loss_avg:.4f})")
        
        if valid_loss_avg < best_valid_loss:
            best_valid_loss, best_epoch = valid_loss_avg, epoch
            
            # 1. 创建符合平台规范的目录 (您的代码已正确实现)
            save_dir = Path(ckpt_path, f"global_step_{global_step}_epoch_{epoch}_loss_{valid_loss_avg:.4f}")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 2. 保存模型到该目录中
            model_path = save_dir / "model.pt"
            torch.save(model.state_dict(), model_path)
            
            print(f"🎉 New best model at epoch {best_epoch}. Saved to: {model_path}")
            
            # 3.在ckpt_path根目录下创建一个符号链接，指向新的最佳模型
            symlink_path = ckpt_path / "model.pt"
            if symlink_path.is_symlink():
                print(f"   Removing old symlink: {symlink_path}")
                symlink_path.unlink()
            elif symlink_path.exists(): # 如果是一个实体文件，也删掉
                print(f"   Removing old file: {symlink_path}")
                symlink_path.unlink()
                
            os.symlink(model_path, symlink_path)
            print(f"   Created symlink for inference: {symlink_path} -> {model_path}")

            # 4. 删除上一个最佳模型的目录以节省空间
            if last_best_ckpt_dir and last_best_ckpt_dir.exists():
                print(f"   Removing old best checkpoint directory: {last_best_ckpt_dir}")
                shutil.rmtree(last_best_ckpt_dir)
            
            last_best_ckpt_dir = save_dir
            
        else:
            print(f"Validation loss did not improve. Best so far: {best_valid_loss:.4f} at epoch {best_epoch}.")
        
        print("-" * 80) 

    print("="*30, f"\nDone training. Best model from epoch {best_epoch} saved at {last_best_ckpt_dir}", "="*30, sep='\n')
    writer.close(); log_file.close()