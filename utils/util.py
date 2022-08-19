import torch
import random
import logging
import numpy as np
from utils.category_id_map import lv2id_to_lv1id
from sklearn.metrics import f1_score, accuracy_score
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

def setup_device(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()
def setup_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def setup_logging(args):
    log_name =args.log_save_path + f'{args.model_name}.log'
    logging.basicConfig(filename=log_name,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    return logger


def build_optimizer(args, model, num_total_steps, pretrained_model=None,model_lr={'others':5e-4, 'roberta':5e-5}):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    if args.opti == 'AdamW':
        optimizer_grouped_parameters = []
        for layer_name in model_lr:
            lr = model_lr[layer_name]
            if layer_name != 'others':  # 设定了特定 lr 的 layer
                optimizer_grouped_parameters += [
                    {
                        "params": [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay) 
                                                                            and layer_name in n)],
                        "weight_decay": args.weight_decay,
                        "lr": lr,
                    },
                    {
                        "params": [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay) 
                                                                            and layer_name in n)],
                        "weight_decay": 0.0,
                        "lr": lr,
                    },
                ]
            else:  # 其他，默认学习率
                optimizer_grouped_parameters += [
                    {
                        "params": [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay) 
                                                                            and not any(name in n for name in model_lr))],
                        "weight_decay": args.weight_decay,
                        "lr": lr,
                    },
                    {
                        "params": [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay) 
                                                                            and not any(name in n for name in model_lr))],
                        "weight_decay": 0.0,
                        "lr": lr,
                    },
                ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=args.weight_decay)
    warmup_steps = int(num_total_steps*args.warmup_rate)
    if args.sched == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=num_total_steps, num_warmup_steps=warmup_steps)
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,num_training_steps=num_total_steps)
    return optimizer, scheduler

def build_optimizer_clr(args, model, num_total_steps):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    warmup_steps = int(num_total_steps*args.warmup_rate)
    if args.sched == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=num_total_steps, num_warmup_steps=warmup_steps)
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,num_training_steps=num_total_steps)
    return optimizer, scheduler


def build_optimizer_clip(args, model_text, model_image, model_clip, num_total_steps):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model_text.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            "lr": args.bert_learning_rate
        },
        {
            "params": [p for n, p in model_text.named_parameters() if any(nd in n for nd in no_decay)], 
            "weight_decay": 0.0, 
            "lr": args.bert_learning_rate
        },
        {
            "params": [p for n, p in model_image.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            "lr": args.visual_learning_rate
        },
        {
            "params": [p for n, p in model_image.named_parameters() if any(nd in n for nd in no_decay)], 
            "weight_decay": 0.0, 
            "lr": args.visual_learning_rate
        },
        {
            "params": [p for n, p in model_clip.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            "lr": args.clip_learning_rate
        },
        {
            "params": [p for n, p in model_clip.named_parameters() if any(nd in n for nd in no_decay)], 
            "weight_decay": 0.0, 
            "lr": args.clip_learning_rate
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    warmup_steps = int(num_total_steps*args.warmup_rate)
    if args.sched == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=num_total_steps, num_warmup_steps=warmup_steps)
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,num_training_steps=num_total_steps)
    return optimizer, scheduler

def evaluate(predictions, labels):
    # prediction and labels are all level-2 class ids

    lv1_preds = [lv2id_to_lv1id(lv2id) for lv2id in predictions]
    lv1_labels = [lv2id_to_lv1id(lv2id) for lv2id in labels]

    lv2_f1_micro = f1_score(labels, predictions, average='micro')
    lv2_f1_macro = f1_score(labels, predictions, average='macro')
    lv1_f1_micro = f1_score(lv1_labels, lv1_preds, average='micro')
    lv1_f1_macro = f1_score(lv1_labels, lv1_preds, average='macro')
    mean_f1 = (lv2_f1_macro + lv1_f1_macro + lv1_f1_micro + lv2_f1_micro) / 4.0

    eval_results = {'lv1_acc': accuracy_score(lv1_labels, lv1_preds),
                    'lv2_acc': accuracy_score(labels, predictions),
                    'lv1_f1_micro': lv1_f1_micro,
                    'lv1_f1_macro': lv1_f1_macro,
                    'lv2_f1_micro': lv2_f1_micro,
                    'lv2_f1_macro': lv2_f1_macro,
                    'mean_f1': mean_f1}

    return eval_results
