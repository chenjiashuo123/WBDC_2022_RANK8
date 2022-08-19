
import os
import time
import torch
import logging
import warnings
from config import parse_args
from utils.ema_util import EMA
from models.model_2 import MULTIBERT2
from utils.adv_util import FGM, PGD, AWP
from dataset.dataset_helper import create_dataloaders
from utils.util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate

warnings.filterwarnings("ignore")


def validate(model, val_dataloader, ema, epoch, final_save_path):
    ema.apply_shadow()
    model.eval()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            loss, _, pred_label_id, label = model(batch)
            loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(label.cpu().numpy())
            losses.append(loss.cpu().numpy())
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)
    results = {k: round(v, 4) for k, v in results.items()}
    mean_f1 = results['mean_f1']
    torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict(), 'mean_f1': mean_f1},
            f'{final_save_path}/model_epoch_{epoch}_mean_f1_{mean_f1}.bin')
    model.train()
    ema.restore()
    return loss, results


def train_and_validate(args):
    # 1. load data
    train_dataloader, val_dataloader = create_dataloaders(args)
    logging.info(f'Load dataset from {args.train_idx_path}')
    # 2. build model and optimizers
    model = MULTIBERT2(args)
    pretrained_model = None
    num_total_steps = len(train_dataloader) * args.max_epochs
    warmup_steps = int(num_total_steps*args.warmup_rate)
    optimizer, scheduler = build_optimizer(args, model, num_total_steps, pretrained_model, model_lr={'others':5e-5, 'newfc_tag':5e-4, 'visual_backbone':5e-6})
    logging.info(f'Total train steps={num_total_steps}, warmup steps={warmup_steps}')
    logging.info(optimizer)
    logging.info(scheduler)
    if torch.cuda.is_available():
        logging.info(f"Use GPU cuda:{args.n_gpu}")
        model = torch.nn.parallel.DataParallel(model.to(args.device))
    if args.ema_decay > 0:
        ema = EMA(model, args.ema_decay)
        ema.register()
    if args.adv_type == 'fgm':
        fgm = FGM(model)
    elif args.adv_type == 'pgd':
        pgd = PGD(model)
        K = 3
    elif args.adv_type == 'awp':
        awp = AWP(model)
        K = 3
    # 3. training
    step = 0
    best_score = args.best_score
    best_step = 0
    start_time = time.time()
    savedmodel_path = args.savedmodel_path
    save_model_name = f'{args.model_name}'
    final_save_path = savedmodel_path + save_model_name
    os.makedirs(final_save_path, exist_ok=True)
    for epoch in range(args.max_epochs):
        for batch in train_dataloader:
            model.train()
            loss, accuracy, _, _ = model(batch)
            loss = loss.mean()
            loss.backward()
            accuracy = accuracy.mean()
            # 对抗训练
            if args.adv_type == 'fgm':
                fgm.attack()
                loss_adv, _, _, _ = model(batch)
                loss_adv = loss_adv.mean()
                loss_adv.backward() 
                fgm.restore()
            elif args.adv_type == 'pgd':
                pgd.backup_grad()
                # 对抗训练
                for t in range(K):
                    pgd.attack(is_first_attack=(t==0)) #在embedding上添加对抗扰动, first attack时备份param.data
                    if t != K-1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    loss_adv, _, _, _ = model(batch)
                    loss_adv = loss_adv.mean()
                    loss_adv.backward() 
                pgd.restore() # 恢复embedding参数
            elif args.adv_type == 'awp':
                awp.backup_grad()
                # 对抗训练
                for t in range(K):
                    awp.attack(is_first_attack=(t==0)) #在embedding上添加对抗扰动, first attack时备份param.data
                    if t != K-1:
                        model.zero_grad()
                    else:
                        awp.restore_grad()
                    loss_adv, _, _, _ = model(batch)
                    loss_adv = loss_adv.mean()
                    loss_adv.backward() 
                awp.restore() # 恢复embedding参数
            optimizer.step()
            if args.ema_decay > 0:
                ema.update()
            optimizer.zero_grad()
            scheduler.step()
            step += 1
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, accuracy {accuracy:.3f}")
            if args.validation_steps > 0:
                if step % args.validation_steps == 0:
                    if args.ema_decay > 0:
                        ema.apply_shadow()
                    model.eval()
                    predictions = []
                    labels = []
                    losses = []
                    with torch.no_grad():
                        for batch in val_dataloader:
                            loss, _, pred_label_id, label = model(batch)
                            loss = loss.mean()
                            predictions.extend(pred_label_id.cpu().numpy())
                            labels.extend(label.cpu().numpy())
                            losses.append(loss.cpu().numpy())
                    loss = sum(losses) / len(losses)
                    results = evaluate(predictions, labels)
                    results = {k: round(v, 4) for k, v in results.items()}
                    logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")
                    mean_f1 = results['mean_f1']
                    if mean_f1 > best_score:
                        best_score = mean_f1
                        best_step = step
                        torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict(), 'mean_f1': mean_f1},
                                f'{final_save_path}/best-model.bin')
                    model.train()
                    if args.ema_decay > 0:
                        ema.restore()
        if args.ema_decay > 0:
            ema.apply_shadow()
        model.eval()
        predictions = []
        labels = []
        losses = []
        with torch.no_grad():
            for batch in val_dataloader:
                loss, _, pred_label_id, label = model(batch)
                loss = loss.mean()
                predictions.extend(pred_label_id.cpu().numpy())
                labels.extend(label.cpu().numpy())
                losses.append(loss.cpu().numpy())
        loss = sum(losses) / len(losses)
        results = evaluate(predictions, labels)
        results = {k: round(v, 4) for k, v in results.items()}
        logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")
        mean_f1 = results['mean_f1']
        if mean_f1 > best_score:
            best_score = mean_f1
            best_step = step
            torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict(), 'mean_f1': mean_f1},
                    f'{final_save_path}/best-model.bin')
            torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict(), 'mean_f1': mean_f1},
                f'{final_save_path}/model_epoch_{epoch}.bin')
        model.train()
        if args.ema_decay > 0:
            ema.restore()
    logging.info(f"Best score：{best_score} at step {best_step}")
def main():
    args = parse_args()
    setup_device(args)
    setup_seed(args)
    args.train_idx_path = f"{args.train_val_dir}/train_idx_{args.fold}.npy"
    args.val_idx_path = f"{args.train_val_dir}/validation_idx_{args.fold}.npy"
    setup_logging(args)
    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)
    train_and_validate(args)


if __name__ == '__main__':
    main()
