import os
import json
import timm
import torch
import random
import zipfile
import numpy as np
from PIL import Image
from io import BytesIO
from os import truncate
from transformers import BertTokenizer
from transformers import CLIPProcessor, CLIPModel, CLIPConfig,CLIPVisionModel
from utils.category_id_map import category_id_to_lv2id, category_id_to_lv1id
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor, ColorJitter, RandomHorizontalFlip, RandomVerticalFlip, RandomCrop, RandomAffine, RandomRotation




def create_dataloaders(args):
    dataset = MultiModalDataset(args, args.train_annotation, args.train_zip_feats)
    size = len(dataset)
    if args.full:
        train_idx = [i for i in range(size)]
        validation_idx = np.load(args.val_idx_path).tolist()
    else:
        train_idx = np.load(args.train_idx_path).tolist()
        validation_idx = np.load(args.val_idx_path).tolist()
    train_dataset, val_dataset = torch.utils.data.Subset(dataset, train_idx), torch.utils.data.Subset(dataset, validation_idx)
    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  drop_last=True,
                                  pin_memory=True,
                                  num_workers=args.num_workers,
                                  prefetch_factor=args.prefetch)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.val_batch_size,
                                sampler=val_sampler,
                                drop_last=False,
                                pin_memory=True,
                                num_workers=args.num_workers,
                                prefetch_factor=args.prefetch)
    return train_dataloader, val_dataloader

def create_dataloaders_aug(args):
    transforms = None
    if args.video_aug:
          transforms = Compose([
              # RandomRotation(degrees=args.RandomRotation, expand=True),
              Resize(256),
              # ColorJitter(brightness=args.ColorJitter, contrast=args.ColorJitter,
              #                        saturation=args.ColorJitter),  # ??????1
              RandomCrop(224),
              RandomHorizontalFlip(),
              RandomVerticalFlip(),
              ToTensor(),
              Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
          ])
    train_dataset = MultiModalDataset(args, args.train_annotation, args.train_zip_feats, transform=transforms)
    val_dataset = MultiModalDataset(args, args.val_annotation, args.train_zip_feats, val_mode=True)
    size = len(train_dataset)
    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  drop_last=True,
                                  pin_memory=True,
                                  num_workers=args.num_workers,
                                  prefetch_factor=args.prefetch)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.val_batch_size,
                                sampler=val_sampler,
                                drop_last=False,
                                pin_memory=True,
                                num_workers=args.num_workers,
                                prefetch_factor=args.prefetch)
    return train_dataloader, val_dataloader

def create_dataloaders_pretrain(args):
    dataset = MultiModalDataset(args, args.pretrain_annotation, args.pretrain_zip_feats, test_mode=True)
    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  drop_last=True,
                                  pin_memory=True,
                                  num_workers=args.num_workers,
                                  prefetch_factor=args.prefetch)
    return train_dataloader

class MultiModalDataset(Dataset):
    """ A simple class that supports multi-modal inputs.

    For the visual features, this dataset class will read the pre-extracted
    features from the .npy files. For the title information, it
    uses the BERT tokenizer to tokenize. We simply ignore the ASR & OCR text in this implementation.

    Args:
        ann_path (str): annotation file path, with the '.json' suffix.
        zip_feats (str): visual feature zip file path.
        test_mode (bool): if it's for testing.
    """

    def __init__(self,
                 args,
                 ann_path: str,
                 zip_feats: str,
                 test_mode: bool = False,
                 val_mode: bool = False,
                 transform = None,
                 fp16 = False):
        self.max_frame = args.max_frames
        self.bert_seq_length = args.bert_seq_length
        self.test_mode = test_mode
        self.val_mode = val_mode

        # lazy initialization for zip_handler to avoid multiprocessing-reading error
        self.zip_feat_path = zip_feats
        self.zip_frame_dir = args.train_zip_frames
        self.handles = [None for _ in range(args.num_workers)]
        self.bert_input = args.bert_input
        self.trunction = args.trunction_mode
        self.pretrain = args.pretrain
        self.fp16 = fp16


        # load annotations
        with open(ann_path, 'r', encoding='utf8') as f:
            self.anns = json.load(f)

        # initialize the text tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir, use_fast=True, cache_dir=args.bert_cache)
        if transform is not None:
            self.transform = transform
        else:
            self.transform = Compose([
                    Resize(256),
                    CenterCrop(224),
                    ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
                ])

    def __len__(self) -> int:
        return len(self.anns)

    def get_visual_feats(self, worker_id, idx: int) -> tuple:
        # read data from zipfile
        vid = self.anns[idx]['id']
        if self.handles[worker_id] is None:
            self.handles[worker_id] = zipfile.ZipFile(self.zip_feat_path, 'r')
        raw_feats = np.load(BytesIO(self.handles[worker_id].read(name=f'{vid}.npy')), allow_pickle=True)
        raw_feats = raw_feats.astype(np.float32)  # float16 to float32
        num_frames, feat_dim = raw_feats.shape

        feat = np.zeros((self.max_frame, feat_dim), dtype=np.float32)
        mask = np.ones((self.max_frame,), dtype=np.int32)
        if num_frames <= self.max_frame:
            feat[:num_frames] = raw_feats
            mask[num_frames:] = 0
        else:
            # if the number of frames exceeds the limitation, we need to sample
            # the frames.
            if self.test_mode or self.val_mode:
                # uniformly sample when test mode is True
                step = num_frames // self.max_frame
                select_inds = list(range(0, num_frames, step))
                select_inds = select_inds[:self.max_frame]
            else:
                # randomly sample when test mode is False
                select_inds = list(range(num_frames))
                random.shuffle(select_inds)
                select_inds = select_inds[:self.max_frame]
                select_inds = sorted(select_inds)
            for i, j in enumerate(select_inds):
                feat[i] = raw_feats[j]
        feat = torch.FloatTensor(feat)
        mask = torch.LongTensor(mask)
        return feat, mask
    
    def get_visual_frames(self, idx: int) -> tuple:
        # read data from zipfile
        vid = self.anns[idx]['id']
        zip_path = os.path.join(self.zip_frame_dir, f'{vid[-3:]}/{vid}.zip')
        handler = zipfile.ZipFile(zip_path, 'r')
        namelist = sorted(handler.namelist())
        num_frames = len(namelist)
        frame = torch.zeros((self.max_frame, 3, 224, 224), dtype=torch.float32)
        mask = torch.zeros((self.max_frame, ), dtype=torch.long)
        if num_frames <= self.max_frame:
            # load all frame
            select_inds = list(range(num_frames))
        else:
            # if the number of frames exceeds the limitation, we need to sample
            # the frames.
            if self.test_mode:
                # uniformly sample when test mode is True
                step = num_frames // self.max_frame
                select_inds = list(range(0, num_frames, step))
                select_inds = select_inds[:self.max_frame]
            else:
                # randomly sample when test mode is False
                select_inds = list(range(num_frames))
                random.shuffle(select_inds)
                select_inds = select_inds[:self.max_frame]
                select_inds = sorted(select_inds)
        for i, j in enumerate(select_inds):
            mask[i] = 1
            img_content = handler.read(namelist[j])
            img = Image.open(BytesIO(img_content))
            img_tensor = self.transform(img)
            frame[i] = img_tensor
        return frame, mask

    def tokenize_text_token(self, title_token, ocr_token, asr_token) -> tuple:
        tokens = ["[CLS]"]+ title_token + ["[SEP]"] +ocr_token  + ["[SEP]"] + asr_token + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding_length = self.bert_seq_length - len(input_ids)
        input_mask += ([0] * padding_length)
        input_ids += ([0] * padding_length)
        input_ids = torch.LongTensor(input_ids)
        mask = torch.LongTensor(input_mask)
        return input_ids, mask
    def truncate_text_sequence(self, title_token, asr_token, ocr_token, token_len):
        while True:
            total_length = len(title_token) + len(asr_token) + len(ocr_token)
            if total_length <= token_len:
                break
            if len(title_token) > len(asr_token):
                if len(title_token) > len(ocr_token):
                    title_token.pop()
                elif len(asr_token) > len(ocr_token):
                    asr_token.pop()
                else:
                    ocr_token.pop()
            else:
                if len(asr_token) > len(ocr_token):
                    asr_token.pop()
                elif len(title_token) > len(ocr_token):
                    title_token.pop()
                else:
                    ocr_token.pop()
    def tokenize_text_cat(self, text: str) -> tuple:
        encoded_inputs = self.tokenizer(text, max_length=self.bert_seq_length, padding='max_length', truncation=True)
        position_ids = [i for i in range(self.bert_seq_length)]
        segment_ids = [0] * self.bert_seq_length
        input_ids = torch.LongTensor(encoded_inputs['input_ids'])
        mask = torch.LongTensor(encoded_inputs['attention_mask'])
        return input_ids, mask
    def __getitem__(self, idx: int) -> dict:
        # Step 1, load visual features from zipfile.
        worker_info = torch.utils.data.get_worker_info()
        # end2end training: get raw frame
        if self.e2e_training:
            frame_input, frame_mask = self.get_visual_frames(idx)
        else:
            frame_input, frame_mask = self.get_visual_feats(worker_info.id, idx)
        # ??????fp16??????
        if self.fp16:
            frame_input = frame_input.half()
        # Step 2, load title tokens
        ocr_text_list = self.anns[idx]['ocr']
        ocr_text = ''
        for item in ocr_text_list:
            ocr_text += item['text']
        title = self.anns[idx]['title']
        asr_text = self.anns[idx]['asr']
        if self.bert_input == 'title':
            title_input, title_mask = self.tokenize_text_cat(self.anns[idx]['title'])
        else:
            segment_len = self.bert_seq_length - 4
            title_token = self.tokenizer.tokenize(title)
            asr_token = self.tokenizer.tokenize(asr_text)
            ocr_token = self.tokenizer.tokenize(ocr_text)
            self.truncate_text_sequence(title_token, asr_token, ocr_token, segment_len)
            title_input, title_mask= self.tokenize_text_token(title_token, ocr_token, asr_token)
        # Step 3, summarize into a dictionary
        data = dict(
            frame_input=frame_input,
            frame_mask=frame_mask,
            title_input=title_input,
            title_mask=title_mask
        )
        # ?????????????????????vtc??????
        if self.pretrain:
            asr_ocr_input_ids, asr_ocr_mask = self.tokenize_text_cat(ocr_text + asr_text)
            data['asr_ocr_input_ids'] = asr_ocr_input_ids
            data['asr_ocr_mask'] = asr_ocr_mask
        # Step 4, load label if not test mode
        if not self.test_mode:
            label = category_id_to_lv2id(self.anns[idx]['category_id'])
            data['label'] = torch.LongTensor([label])

        return data
