import os
import time
import torch
import logging
import warnings
from config import parse_args
from tkinter.messagebox import NO
from transformers import BertModel
from models.model_clip import ClipModel
from transformers import CLIPVisionModel
from dataset.dataset_helper import create_dataloaders_pretrain
from utils.util import setup_device, setup_seed, setup_logging, build_optimizer_clip
warnings.filterwarnings("ignore")

def train_and_validate(args):
    # 1. load data
    train_dataloader = create_dataloaders_pretrain(args)
    logging.info(f'Load dataset from {args.pretrain_zip_frames}')
    # 2. build model and optimizers
    model_text = BertModel.from_pretrained(args.bert_path)
    model_image = CLIPVisionModel.from_pretrained(args.clip_pretrained_path)
    model_clip = ClipModel(args)
    
    num_total_steps = len(train_dataloader) * args.max_epochs
    warmup_steps = int(num_total_steps*args.warmup_rate)
    pretrained_model = None
    optimizer, scheduler = build_optimizer_clip(args, model_text, model_image, model_clip, num_total_steps)
    step = 0
    # 断点恢复
    start_epoch = -1
    if args.resume:
        if os.path.isfile(args.resume_checkpoint):
            checkpoint = torch.load(args.resume_checkpoint)  # 加载断点
            logging.info(f'resume pretrain weight from {args.resume_checkpoint}')
            model_image.load_state_dict(checkpoint['model_image'])  # 加载模型可学习参数
            model_clip.load_state_dict(checkpoint['model_clip'])  # 加载模型可学习参数
            model_text.load_state_dict(checkpoint['model_text'])  # 加载模型可学习参数
            optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
            start_epoch = checkpoint['epoch']  # 设置开始的epoch
            step = checkpoint['step']
            scheduler.load_state_dict(checkpoint['scheduler'])#加载lr_scheduler
            del checkpoint
        else:
            logging.info(f'no checkpoint find at {args.resume_checkpoint}')
    logging.info(f'Total train steps={num_total_steps}, warmup steps={warmup_steps}')
    logging.info(optimizer)
    logging.info(scheduler)
    if torch.cuda.is_available():
        logging.info(f"Use GPU cuda:{args.n_gpu}")
        model_image = torch.nn.parallel.DataParallel(model_image.to(args.device))
        model_text = torch.nn.parallel.DataParallel(model_text.to(args.device))
        model_clip = torch.nn.parallel.DataParallel(model_clip.to(args.device))
    # 3. training
    start_time = time.time()
    savedmodel_path = args.savedmodel_path
    save_model_name = f'{args.model_name}'
    final_save_path = savedmodel_path + save_model_name
    os.makedirs(final_save_path, exist_ok=True)
    step_loss = 0
    step_loss_img = 0
    step_loss_text = 0
    
    for epoch in range(start_epoch+1, args.max_epochs):
        model_image.train()
        model_text.train()
        model_clip.train()
        for batch in train_dataloader:
            
            frame_input=batch['frame_input'].to(args.device)
            frame_mask=batch['frame_mask'].to(args.device)
            title_input=batch['title_input'].to(args.device)
            title_mask=batch['title_mask'].to(args.device)
            
            B, N, C, H, W = frame_input.shape
            output_shape = (B, N, -1)
            frame_input = frame_input.view(B * N, C, H, W)

            image_embedding = model_image(frame_input)['pooler_output'].view(*output_shape)
            text_embedding = model_text(title_input, title_mask)['last_hidden_state']
            loss, loss_image, loss_text = model_clip(image_embedding,  text_embedding)
            
            loss = loss.mean()
            loss_image = loss_image.mean()
            loss_text = loss_text.mean()
            
            step_loss += loss
            step_loss_img += loss_image
            step_loss_text += loss_text
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
                step_loss_img = step_loss_img / args.print_steps
                step_loss_text = step_loss_text / args.print_steps
                logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {step_loss:.3f}: loss_img {step_loss_img:.3f}: loss {step_loss_text:.3f}")
                step_loss = 0
                step_loss_img = 0
                step_loss_text = 0
        torch.save({'epoch': epoch, 
            'model_image': model_image.module.state_dict(),
            'model_text': model_text.module.state_dict(), 
            'model_clip': model_clip.module.state_dict(), 
            'step': step, 
            'optimizer': optimizer.state_dict(), 
            'scheduler': scheduler.state_dict()}, f'{final_save_path}/model_epoch_{epoch}.bin')
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
