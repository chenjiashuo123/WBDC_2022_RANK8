import os
import torch
import numpy as np
from config import parse_args
import torch.nn.functional as F
from models.model_1 import MULTIBERT
from models.model_2 import MULTIBERT2
from dataset.dataset_e2e import MultiModalDataset
from utils.category_id_map import lv2id_to_category_id
from torch.utils.data import SequentialSampler, DataLoader


def inference():
    args = parse_args()
    # 1. load data
    dataset = MultiModalDataset(args, args.test_annotation, args.pretrain_zip_frames, test_mode=True)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch)

    # 2. load model
    model_1 = MULTIBERT(args)
    checkpoint = torch.load(args.ckpt_file_1, map_location='cpu')
    model_1.load_state_dict(checkpoint['model_state_dict'])
    if args.fp16:
        model_1 = model_1.half()
    
    model_2 = MULTIBERT(args)
    checkpoint = torch.load(args.ckpt_file_2, map_location='cpu')
    model_2.load_state_dict(checkpoint['model_state_dict'])
    if args.fp16:
        model_2 = model_2.half()
    
    model_3 = MULTIBERT2(args)
    checkpoint = torch.load(args.ckpt_file_3, map_location='cpu')
    model_3.load_state_dict(checkpoint['model_state_dict'])
    if args.fp16:
        model_3 = model_3.half()
    model_4 = MULTIBERT(args)
    checkpoint = torch.load(args.ckpt_file_4, map_location='cpu')
    model_4.load_state_dict(checkpoint['model_state_dict'])
    if args.fp16:
        model_4 = model_4.half()
    if torch.cuda.is_available():
        model_1 = torch.nn.parallel.DataParallel(model_1.cuda())
        model_2 = torch.nn.parallel.DataParallel(model_2.cuda())
        model_3 = torch.nn.parallel.DataParallel(model_3.cuda())
        model_4 = torch.nn.parallel.DataParallel(model_4.cuda())
    model_1.eval()
    model_2.eval()
    model_3.eval()
    model_4.eval()
    # 3. inference
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            _,logit_1 = model_1(batch, inference=True)
            _,logit_2 = model_2(batch, inference=True)
            _,logit_3 = model_3(batch, inference=True, fp16=args.fp16)
            _,logit_4 = model_4(batch, inference=True)
            logit_1 =F.softmax(logit_1, 1)
            logit_2 =F.softmax(logit_2, 1)
            logit_3 =F.softmax(logit_3, 1)
            logit_4 =F.softmax(logit_4, 1)
            logits = logit_1 + logit_2 + 0.7*logit_3 + logit_4
            pred_label_id = torch.argmax(logits, dim=1)
            predictions.extend(pred_label_id.cpu().numpy())

    # 4. dump results
    print(f"Save result to {args.test_output_csv}")
    with open(args.test_output_csv, 'w') as f:
        for pred_label_id, ann in zip(predictions, dataset.anns):
            video_id = ann['id']
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')

if __name__ == '__main__':
    # inference_logits()
    inference()