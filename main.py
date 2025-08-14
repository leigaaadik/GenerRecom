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
from torch.optim.lr_scheduler import LinearLR

from dataset import MyDataset
from model import BaselineModel


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--maxlen', default=101, type=int)
    parser.add_argument('--warmup_steps', default=2000, type=int, help="Number of warmup steps for learning rate scheduler")
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--weight_decay', default=0.01, type=float, help="Weight decay for AdamW optimizer")
    
    parser.add_argument('--hidden_units', default=128, type=int)
    parser.add_argument('--num_blocks', default=4, type=int)
    parser.add_argument('--num_heads', default=4, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])
    
    parser.add_argument('--use_triplet_loss', default=True, action=argparse.BooleanOptionalAction, help="Enable Triplet Loss in addition to InfoNCE")
    parser.add_argument('--triplet_loss_margin', default=1.0, type=float, help="Margin for the Triplet Loss")
    parser.add_argument('--infonce_loss_weight', default=0.95, type=float, help="Weight for the InfoNCE loss component")

    parser.add_argument('--clip_grad_norm', default=1.0, type=float, help="Max norm for gradient clipping, set to 0 to disable")

    args = parser.parse_args()
    return args

def evaluate(model, dataloader, device, k_list=[1, 10]):
    model.eval()
    
    all_ranks = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Metrics"):
            seq, pos, neg, token_type, next_token_type, _, seq_feat, pos_feat, neg_feat = batch
            
            valid_indices = (next_token_type == 1).nonzero(as_tuple=True)
            if valid_indices[0].numel() == 0:
                continue

            seq, pos, neg, token_type = seq.to(device), pos.to(device), neg.to(device), token_type.to(device)

            log_feats = model.log2feats(seq, token_type, seq_feat)
            pos_embs = model.feat2emb(pos, pos_feat, include_user=False)
            neg_embs = model.feat2emb(neg, neg_feat, include_user=False)
            
            log_feats_valid = log_feats[valid_indices]
            pos_embs_valid = pos_embs[valid_indices]
            neg_embs_valid = neg_embs[valid_indices]
            
            log_feats_valid = log_feats_valid / log_feats_valid.norm(dim=-1, keepdim=True)
            pos_embs_valid = pos_embs_valid / pos_embs_valid.norm(dim=-1, keepdim=True)
            neg_embs_valid = neg_embs_valid / neg_embs_valid.norm(dim=-1, keepdim=True)

            pos_scores = (log_feats_valid * pos_embs_valid).sum(dim=-1)

            candidates = torch.cat([pos_embs_valid, neg_embs_valid], dim=0)
            all_scores = log_feats_valid @ candidates.t()
            
            ranks = (all_scores > pos_scores.unsqueeze(1)).sum(dim=1) + 1
            all_ranks.append(ranks)

    if not all_ranks:
        return {f'{metric}@{k}': 0.0 for k in k_list for metric in ['hr', 'ndcg']}

    all_ranks = torch.cat(all_ranks).cpu().numpy()
    metrics = {}
    for k in k_list:
        hr_k = (all_ranks <= k).mean()
        metrics[f'hr@{k}'] = hr_k
        
        in_top_k = all_ranks <= k
        ndcg_k = (1.0 / np.log2(all_ranks[in_top_k] + 1)).sum() / len(all_ranks)
        metrics[f'ndcg@{k}'] = ndcg_k
        
    return metrics

def get_scheduler(optimizer, args, total_steps):
    def lr_lambda(current_step):
        if current_step < args.warmup_steps:
            return float(current_step) / float(max(1, args.warmup_steps))
        
        progress = float(current_step - args.warmup_steps) / float(max(1, total_steps - args.warmup_steps))
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

if __name__ == '__main__':
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    
    data_path = os.environ.get('TRAIN_DATA_PATH')

    args = get_args()
    dataset = MyDataset(data_path, args)
    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn
    )
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    model.user_emb.weight.data[0, :] = 0

    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0

    epoch_start_idx = 1

    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path.split('global_step')[0]
            if 'epoch=' in tail:
                 epoch_str = tail[tail.find('epoch=') + 6:]
                 epoch_start_idx = int(epoch_str[: epoch_str.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)

    num_train_steps_per_epoch = len(train_loader)
    total_train_steps = args.num_epochs * num_train_steps_per_epoch
    scheduler = get_scheduler(optimizer, args, total_train_steps)
    
    global_step = 0
    print("Start training")
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        if args.inference_only:
            break
        for step, batch in tqdm(enumerate(train_loader), total=num_train_steps_per_epoch, desc=f"Training Epoch {epoch}"):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            seq, pos, neg = seq.to(args.device), pos.to(args.device), neg.to(args.device)
            
            optimizer.zero_grad()
            loss = model(
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
            )
            
            if loss.item() > 0:
                log_json = json.dumps(
                    {'global_step': global_step, 'loss': loss.item(), 'epoch': epoch, 'time': time.time()}
                )
                log_file.write(log_json + '\n')
                log_file.flush()
                if global_step % 100 == 0:
                    print(log_json)

                writer.add_scalar('Loss/train', loss.item(), global_step)
                
                loss.backward()

                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
                
                optimizer.step()

            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('LearningRate/train', current_lr, global_step)
            scheduler.step()
            global_step += 1
            
        model.eval()
        valid_loss_sum = 0
        num_valid_batches = 0
        with torch.no_grad():
            for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader), desc="Calculating Validation Loss"):
                seq, pos, neg, token_type, next_token_type, _, seq_feat, pos_feat, neg_feat = batch
                seq, pos, neg = seq.to(args.device), pos.to(args.device), neg.to(args.device)

                loss = model(
                    seq, pos, neg, token_type, next_token_type, None, seq_feat, pos_feat, neg_feat
                )
                if loss.item() > 0:
                    valid_loss_sum += loss.item()
                    num_valid_batches += 1
        
        valid_loss_avg = valid_loss_sum / num_valid_batches if num_valid_batches > 0 else 0
        writer.add_scalar('Loss/valid', valid_loss_avg, global_step)
        print(f"\nEpoch {epoch} | Valid Loss: {valid_loss_avg:.4f}")

        metrics = evaluate(model, valid_loader, args.device, k_list=[1, 10])
        for metric_name, metric_value in metrics.items():
            print(f"Epoch {epoch} | Valid {metric_name.upper()}: {metric_value:.4f}")
            metric_group, metric_k = metric_name.split('@')
            writer.add_scalar(f'Metrics/{metric_group.upper()}@{metric_k}', metric_value, global_step)
        
        log_metrics = {'global_step': global_step, 'epoch': epoch, 'valid_loss': valid_loss_avg, **metrics}
        log_file.write(json.dumps(log_metrics) + '\n')
        log_file.flush()

        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_avg:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")
        print(f"Model saved to {save_dir}")

    print("Done")
    writer.close()
    log_file.close()