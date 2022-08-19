#%%writefile data/masklm.py
import torch
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers import AutoTokenizer
class MaskLM(object):
    def __init__(self, tokenizer_path='bert-base-chinese', mlm_probability=0.15):
        self.mlm_probability = mlm_probability
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
    def torch_mask_tokens(self, inputs: Any, masks: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        masks[masked_indices] = 0
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, masks, labels

    def torch_mask_tokens_with_pad(self, inputs: Any, mask: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        mask[masked_indices] = 0
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, mask
    
class MaskVideo(object):
    def __init__(self, mlm_probability=0.15):
        self.mlm_probability = mlm_probability
    def torch_mask_frames(self, video_feature, video_mask):
        probability_matrix = torch.full(video_mask.shape, 0.9 * self.mlm_probability)
        probability_matrix = probability_matrix * video_mask
        masked_indices = torch.bernoulli(probability_matrix).bool()
        video_labels_index = torch.arange(video_feature.size(0) * video_feature.size(1)).view(-1, video_feature.size(1))
        video_labels_index = -100 * ~masked_indices + video_labels_index * masked_indices
        # 90% mask video fill all 0.0
        masked_indices_unsqueeze = masked_indices.unsqueeze(-1).expand_as(video_feature)
        inputs = video_feature.data.masked_fill(masked_indices_unsqueeze, 0.0)
        labels = video_feature[masked_indices_unsqueeze].contiguous().view(-1, video_feature.size(2)) 
        return inputs, video_labels_index
    
class ShuffleVideo(object):
    def __init__(self):
        pass
    def torch_shuf_video(self, video_feature, video_mask):
        bs = video_feature.size()[0]
        # batch 内前一半 video 保持原顺序，后一半 video 逆序
        shuf_index = torch.tensor(list(range(bs // 2)) + list(range(bs //2, bs))[::-1])
        # shuf 后的 label
        label = (torch.tensor(list(range(bs))) == shuf_index).float()
        video_feature = video_feature[shuf_index]
        video_mask = video_mask[shuf_index]
        return video_feature, video_mask, label

class MMM_Mask(object):
    def __init__(self, tokenizer_path='bert-base-chinese'):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.mlm_probability = 1.0
    def torch_mask_tokens(self, inputs: Any, mask: Any,special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any, Any]:
        """
        batch 后一半的text全都mask掉
        """
        labels = inputs.clone()
        bs = labels.shape[0]
        mask_bs = bs // 2
        labels_mask = labels[mask_bs:,:]
        mask_mask = mask[mask_bs:,:]
        unmaks_mask = mask[:mask_bs,:]

        pad_tokens_mask = torch.tensor(1-mask_mask, dtype=torch.bool)
        masked_indices = torch.full(labels_mask.shape, 1.0)
        masked_indices.masked_fill_(pad_tokens_mask, value=0.0)
        masked_indices = masked_indices.bool()
        unmasked_indices = torch.full(unmaks_mask.shape, 0.0)
        total_indices = torch.cat((unmasked_indices, masked_indices), dim=0).bool()

        labels_mask[~masked_indices] = -100  # We only compute loss on masked tokens
        inputs[total_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        return inputs, labels[mask_bs:,:]
    def torch_mask_frames(self, video_feature, video_mask):
        """
        batch 前一半的video全都mask掉
        """
        bs = video_mask.shape[0]
        half_bs = bs//2
        video_mask_one = video_mask[:half_bs, :]
        video_mask_two = video_mask[half_bs:, :]
        mask_video_feature = video_feature[:half_bs, :, :]
        masked_indices = torch.full(video_mask_one.shape, 1.0)
        masked_indices = masked_indices * video_mask_one
        masked_indices = masked_indices.bool()
        video_labels_index = torch.arange(mask_video_feature.size(0) * mask_video_feature.size(1)).view(-1, mask_video_feature.size(1))
        video_labels_index = -100 * ~masked_indices + video_labels_index * masked_indices
        unmasked_indices = torch.full(video_mask_two.shape, 0.0)
        total_indices = torch.cat((masked_indices, unmasked_indices), 0).bool()
        total_indices_unsqueeze = total_indices.unsqueeze(-1).expand_as(video_feature)
        inputs = video_feature.data.masked_fill(total_indices_unsqueeze, 0.0)
        labels = video_feature[total_indices_unsqueeze].contiguous().view(-1, video_feature.size(2))
        return inputs, video_labels_index