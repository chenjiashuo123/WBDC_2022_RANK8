import os
import io
import json
import torch
import timm
import zipfile
import argparse
import numpy as np
import torch.nn as nn
from PIL import Image
from transformers import logging
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor
from transformers import CLIPProcessor, CLIPModel, CLIPConfig,CLIPVisionModel, ViTModel
logging.set_verbosity_warning()
logging.set_verbosity_error()
class RawFrameDataset(Dataset):

    def __init__(self,
                 ann_path: str,
                 zip_frame_dir: str,
                 max_video_frames: int = 32):
        """ This class is used to load raw video frames.
        Args:
            ann_paths (str): the annotation file path.
            zip_frame_dir (str): the directory that saves zip frames.
            max_video_frames (str): the maximum number of video frames.
        """
        # load annotations
        with open(ann_path, 'r', encoding='utf8') as f:
            self.anns = json.load(f)
        self.zip_frame_dir = zip_frame_dir
        self.max_video_frames = max_video_frames
        
        # we follow the common practice as in the ImageNet's preprocessing.
        self.transform = Compose([
                Resize(256),
                CenterCrop(224),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self) -> dict:
        return len(self.anns)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """ Extract the frame tensor from zipped file.
        The output tensor is in shape of [MAX_FRAMES, 3, 224, 224]
        """
        feedid = self.anns[idx]['id']
        zip_path = os.path.join(self.zip_frame_dir, feedid[-3:], f'{feedid}.zip')
        handler = zipfile.ZipFile(zip_path, 'r')
        img_name_list = handler.namelist()
        img_name_list = sorted(img_name_list)
        img_name_list = img_name_list[:self.max_video_frames]
        img_tensor = torch.zeros(self.max_video_frames, 3, 224, 224)
        for i, img_name in enumerate(img_name_list):
            i_img_content = handler.read(img_name)
            i_img = Image.open(io.BytesIO(i_img_content))
            i_img_tensor = self.transform(i_img)
            img_tensor[i, ...] = i_img_tensor
        handler.close()
        num_frames = torch.LongTensor([len(img_name_list)])
        return dict(img=img_tensor, num_frames=num_frames)


def parse_args():
    parser = argparse.ArgumentParser("Visual feature extraction")
    parser.add_argument('--zip_frame_dir', type=str, default='/opt/ml/input/data/zip_frames//unlabeled/')
    parser.add_argument('--ann_path', type=str, default='/opt/ml/input/data/annotations/unlabeled.json')
    parser.add_argument('--pretrained_weight', type=str, default='../opensource_models/clip-vit-base-patch32')
    parser.add_argument('--output_path', type=str, default='zip_feats/unlabeled_clip.zip')
    parser.add_argument('--model', type=str, default='clip')
    args = parser.parse_args()
    return args



def build_model_clip(pretrained_weight) -> torch.nn.Module:
    """ Load the pretrianed feature extractor (Swin-T here). """
    print(f"Load pretrained {pretrained_weight}")
    model = CLIPVisionModel.from_pretrained(pretrained_weight)
    if torch.cuda.is_available():
        model = DataParallel(model.cuda(), device_ids=list(range(torch.cuda.device_count())))
    model.eval()
    return model




def main():
    args = parse_args()
    model = build_model_clip(args.pretrained_weight)
    dataset = RawFrameDataset(args.ann_path, args.zip_frame_dir)
    # batch-size == 8 is fine for V100 GPU, please consider use smaller batch-size if OOM issue occurs.
    dataloader = DataLoader(dataset, batch_size=64, num_workers=8, shuffle=False, pin_memory=True, drop_last=False)
    assert not os.path.isfile(args.output_path), f"{args.output_path} already exists. " \
                                                  "If you want to override it, please manually delete this file."
    output_handler = zipfile.ZipFile(args.output_path, 'w', compression=zipfile.ZIP_STORED)
    with torch.no_grad():
        cur = 0
        for dataitem in dataloader:
            img, num_frames = dataitem['img'], dataitem['num_frames']
            B, L = img.shape[0:2]
            img = img.view((B * L, ) + img.shape[2:])
            feature = model(img)['pooler_output']
            feature = feature.view(B, L, -1)
            feature = feature.cpu().numpy().astype(np.float16)
            for i in range(B):
                feedid = dataset.anns[cur]['id']
                ioproxy = io.BytesIO()
                np.save(ioproxy, feature[i, :int(num_frames[i])])
                npy_str = ioproxy.getvalue()
                output_handler.writestr(f'{feedid}.npy', npy_str)
                cur += 1
                if cur % 1000 == 0:
                    print(f"Extract feature {cur}/{len(dataset)}")
    output_handler.close()


if __name__ == '__main__':
    main()
