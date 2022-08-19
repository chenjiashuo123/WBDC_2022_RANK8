import argparse
from tkinter.messagebox import NO


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline for Weixin Challenge 2022")

    parser.add_argument("--seed", type=int, default=802, help="random seed.")
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout ratio')

    # ========================= Data Prepare Configs ==========================
    parser.add_argument('--fold_save_path', type=str, default='fold_data/five/')
    parser.add_argument('--skf_fold', type=int, default=5)

    # ========================= Data Configs ==========================
    parser.add_argument('--train_annotation', type=str, default='/opt/ml/input/data/annotations/labeled.json')
    parser.add_argument('--val_annotation', type=str, default='fold_data/labeled-validation.json')
    parser.add_argument('--test_annotation', type=str, default='/opt/ml/input/data/annotations/test.json')
    parser.add_argument('--train_zip_frames', type=str, default='/opt/ml/input/data/zip_frames/labeled/')
    parser.add_argument('--train_zip_feats', type=str, default='zip_feats/labeled_clip.zip')
    parser.add_argument('--test_zip_frames', type=str, default='/opt/ml/input/data/zip_frames/test/')
    parser.add_argument('--test_output_csv', type=str, default='/opt/ml/output/result.csv')
    
    parser.add_argument('--pretrain_annotation', type=str, default='/opt/ml/input/data/annotations/unlabeled.json')
    parser.add_argument('--pretrain_zip_feats', type=str, default='zip_feats/unlabeled_clip.zip')
    parser.add_argument('--pretrain_zip_frames', type=str, default='/opt/ml/input/data/zip_frames/unlabeled/')
    
    parser.add_argument('--test_zip_feats', type=str, default='../data/zip_feats/test_b.zip')
    parser.add_argument('--logits_output_csv', type=str, default='post_process/logits_files')
    parser.add_argument('--val_ratio', default=0.1, type=float, help='split 10 percentages of training data as validation')
    parser.add_argument('--batch_size', default=64, type=int, help="use for training duration per worker")
    parser.add_argument('--val_batch_size', default=256, type=int, help="use for validation duration per worker")
    parser.add_argument('--test_batch_size', default=512, type=int, help="use for testing duration per worker")
    parser.add_argument('--prefetch', default=16, type=int, help="use for training duration per worker")
    parser.add_argument('--num_workers', default=8, type=int, help="num_workers for dataloaders")
    parser.add_argument('--trunction_mode', default='sequence', type=str, help="text trunction mode")
    parser.add_argument('--train_idx_path', type=str, default='fold_data/ten/train_idx.npy')
    parser.add_argument('--val_idx_path', type=str, default='fold_data/ten/validation_idx.npy')
    parser.add_argument('--train_val_dir', type=str, default='fold_data/five/')


    # ======================== SavedModel Configs =========================
    parser.add_argument('--savedmodel_path', type=str, default='save/')
    parser.add_argument('--model_name', type=str, default='default')
    parser.add_argument('--log_save_path', type=str, default='log/')
    parser.add_argument('--ckpt_file', type=str, default=None)
    parser.add_argument('--best_score', default=0.5, type=float, help='save checkpoint if mean_f1 > best_score')
    parser.add_argument('--logit_save_path', type=str, default='post_process/logits_files/model-1')

    # ========================= Learning Configs ==========================
    parser.add_argument('--max_epochs', type=int, default=5, help='How many epochs')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='clip_grad_norm')
    parser.add_argument('--max_steps', default=50000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--print_steps', type=int, default=100, help="Number of steps to log training metrics.")
    parser.add_argument('--warmup_rate', default=0.1, type=float, help="warm ups for parameters not in bert or vit")
    parser.add_argument('--warmup_steps', default=1000, type=int, help="warm ups for parameters not in bert or vit")
    parser.add_argument('--minimum_lr', default=1e-5, type=float, help='minimum learning rate')
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='initial learning rate')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--loss', type=str, default='cross_entropy')
    parser.add_argument('--adv_type', type=str, default='none')
    parser.add_argument('--mask_word', type=int, default=0)
    parser.add_argument('--full', type=int, default=0)
    parser.add_argument('--validation_steps', type=int, default=0, help="Number of steps to validation.")
    parser.add_argument('--sched', type=str, default='cosine')
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--cooldown_epochs', type=int, default=0)
    parser.add_argument('--decay_rate', type=int, default=1)
    parser.add_argument('--warmup_lr', type=float, default=1e-5)
    parser.add_argument('--noise_lambda', type=float, default=0)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--ema_decay', type=float, default=0)
    parser.add_argument('--opti', type=str, default='AdamW')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--bert_learning_rate', type=float, default=5e-5)
    parser.add_argument('--other_learning_rate', type=float, default=5e-4)
    parser.add_argument('--clip_learning_rate', type=float, default=1e-5)
    parser.add_argument('--visual_learning_rate', type=float, default=5e-6)
    parser.add_argument('--e2e_training', action='store_true', help="end2end train")

    # ========================== Pretrain config =============================
    parser.add_argument('--pretrain_dir', type=str, default='save/pretrain-model/')
    parser.add_argument('--pretrain_load_epoch', type=int, default=-1)
    parser.add_argument('--finetune_load_epoch', type=int, default=-1)
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--visual_clr', action='store_true')
    parser.add_argument('--resume_checkpoint', type=str, default='save/pretrain-clip-fold-0/model_epoch_9.bin')
    parser.add_argument('--init_clip_from_pretrain', action='store_true')
    
    # ========================== inference config =============================
    parser.add_argument('--fp16', action='store_true')
    
    
    # ========================== Clip CLR Pretrain config =============================
    parser.add_argument('--n_views', type=int, default=2)
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')
    parser.add_argument('--out_dim', default=128, type=int,
                        help='feature dimension (default: 128)')

    # ========================== Title BERT =============================
    parser.add_argument('--bert_dir', type=str, default='hfl/chinese-macbert-base')
    parser.add_argument('--bert_path', type=str, default='../opensource_models/chinese-macbert-base')
    parser.add_argument('--pretain_model_dir', type=str, default='bert_ckpt/unibert_pretrain/model_epoch_8.bin')
    parser.add_argument('--bert_input', type=str, default='title_asr_ocr')
    parser.add_argument('--bert_cache', type=str, default='../opensource_models/chinese-macbert-base')
    parser.add_argument('--bert_seq_length', type=int, default=256)
    parser.add_argument('--bert_warmup_steps', type=int, default=5000)
    parser.add_argument('--bert_max_steps', type=int, default=30000)
    parser.add_argument("--bert_hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--cls_layers", type=int, default=-1)
    parser.add_argument("--drop_num", type=int, default=0)


    # ========================== Video =============================
    parser.add_argument('--frame_embedding_size', type=int, default=768)
    parser.add_argument('--max_frames', type=int, default=16)
    parser.add_argument('--vlad_cluster_size', type=int, default=64)
    parser.add_argument('--vlad_groups', type=int, default=8)
    parser.add_argument('--vlad_hidden_size', type=int, default=768, help='nextvlad output size using dense')
    parser.add_argument('--se_ratio', type=int, default=8, help='reduction factor in se context gating')
    parser.add_argument('--video_aug', action='store_true', help='Augmentation video frames')
    # parser.add_argument('--clip_pretrained_path', type=str, default='../opensource_models/clip-vit-base-patch32')
    parser.add_argument('--clip_pretrained_path', type=str, default=None)
    parser.add_argument('--ColorJitter', type=float, default=0.5,help='transforms.ColorJitter')
    parser.add_argument('--RandomRotation', type=int, default=180,help='transforms.RandomRotation')


    # ========================== lxmert config =============================
    parser.add_argument('--albef_conig', type=str, default='configs/albef_pretrain.yaml')
    parser.add_argument('--hidden_size', type=int, default=768, help='Align hidden size')
    parser.add_argument('--video_layers', type=int, default=6, help='video encoder layer')
    parser.add_argument('--mlm_probability', type=float, default=0.15, help='text mask rate')
    parser.add_argument('--cross_num_layers', type=int, default=3, help='number of cross attention layer')

    # ========================== Fusion Layer =============================
    parser.add_argument('--fc_size', type=int, default=512, help="linear size before final linear")


    return parser.parse_args()
