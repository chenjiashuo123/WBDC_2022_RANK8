import os
import time
import torch
import logging
import warnings
from config import parse_args
from tkinter.messagebox import NO
from models.model_1_pretrain import MULTIBERT
from dataset.dataset_helper import create_dataloaders_pretrain
from utils.util import setup_device, setup_seed, setup_logging, build_optimizer
warnings.filterwarnings("ignore")

def train_and_validate(args):
    # 1. load data
    train_dataloader = create_dataloaders_pretrain(args)

    # 2. build model and optimizers
    model = MULTIBERT(args, task=['mlm', 'itm', 'ict', 'mmm', 'mfm'])
    num_total_steps = len(train_dataloader) * args.max_epochs
    warmup_steps = num_total_steps*args.warmup_rate
    logging.info(f'Total train steps={num_total_steps}, warmup steps={warmup_steps}')
    optimizer, scheduler = build_optimizer(args, model, num_total_steps, model_lr={'others':5e-4, 'roberta':1e-4})
    if torch.cuda.is_available():
        logging.info(f"Use GPU cuda:{args.n_gpu}")
        model = torch.nn.parallel.DataParallel(model.to(args.device))

    # 3. training
    step = 0
    start_time = time.time()
    savedmodel_path = args.savedmodel_path
    save_model_name = f'{args.model_name}'
    final_save_path = savedmodel_path + save_model_name
    os.makedirs(final_save_path, exist_ok=True)
    step_loss = 0
    step_mlm_loss = 0
    step_itm_loss = 0
    step_itc_loss = 0
    step_mfm_loss = 0
    step_cap_l_loss = 0
    step_cap_v_loss = 0
    for epoch in range(args.max_epochs):
        for step, batch in enumerate(train_dataloader):
            model.train()
            loss, masked_lm_loss, itm_loss, itc_loss, mfm_loss, cap_l_loss, cap_v_loss = model(batch)
            loss = loss.mean()
            masked_lm_loss = masked_lm_loss.mean()
            itm_loss = itm_loss.mean()
            itc_loss = itc_loss.mean()
            mfm_loss = mfm_loss.mean()
            cap_l_loss = cap_l_loss.mean()
            cap_v_loss = cap_v_loss.mean()
            step_loss += loss
            step_mlm_loss += masked_lm_loss
            step_itm_loss += itm_loss
            step_itc_loss += itc_loss
            step_mfm_loss += mfm_loss
            step_cap_l_loss += cap_l_loss
            step_cap_v_loss += cap_v_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            step += 1
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                step_loss = step_loss / args.print_steps
                step_mlm_loss = step_mlm_loss / args.print_steps
                step_itm_loss = step_itm_loss / args.print_steps
                step_itc_loss = step_itc_loss / args.print_steps
                step_mfm_loss = step_mfm_loss / args.print_steps
                step_cap_l_loss = step_cap_l_loss / args.print_steps
                step_cap_v_loss = step_cap_v_loss / args.print_steps
                logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {step_loss:.3f}, masked_lm_loss {step_mlm_loss:.3f},itm_loss {step_itm_loss:.3f}, itc_loss {step_itc_loss:.3f}, mfm_loss {step_mfm_loss:.3f},cap_l_loss {step_cap_l_loss:.3f}, step_cap_v_loss {step_cap_v_loss:.3f}")
                step_loss = 0
                step_mlm_loss = 0
                step_itm_loss = 0
                step_itc_loss = 0
                step_mfm_loss = 0
                step_cap_l_loss = 0
                step_cap_v_loss = 0
        torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict()}, f'{final_save_path}/model_epoch_{epoch}.bin')
def main():
    args = parse_args()
    setup_logging(args)
    setup_device(args)
    setup_seed(args)
    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)
    train_and_validate(args)
if __name__ == '__main__':
    main()