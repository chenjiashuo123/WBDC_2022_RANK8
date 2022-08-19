import torch
from torch import nn
from transformers import CLIPVisionModel
from dataset.pretrain_helper import MaskLM
from transformers import BertModel, AutoTokenizer, LxmertXLayer
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertConfig


class MULTIBERT2(nn.Module):
    def __init__(self,args, task=['tag']):
        super().__init__()
        
        self.task = set(task)

        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
        self.mlm_probability = args.mlm_probability
        embed_dim = args.hidden_size
        
        if args.clip_pretrained_path is not None:
            print(f'clip weight load from {args.clip_pretrained_path}')
            self.visual_backbone = CLIPVisionModel.from_pretrained(args.clip_pretrained_path)
            unfreeze_layers = ['layers.9','layers.10','layers.11']
            for name ,param in self.visual_backbone.named_parameters():
                param.requires_grad = False
                for ele in unfreeze_layers:
                    if ele in name:
                        param.requires_grad = True
                        break
            for name, param in self.visual_backbone.named_parameters():
                if param.requires_grad:
                    print(name,param.size())
                        
        self.clip_pretrained_path = args.clip_pretrained_path
        vision_width = args.frame_embedding_size
        visual_bert_config = BertConfig.from_pretrained(f'{args.bert_path}/config.json')
        visual_bert_config.num_hidden_layers = args.video_layers

        self.cls_token = nn.Parameter(torch.zeros(1, 1, vision_width))
        # self.visual_fc = nn.Linear(vision_width, vision_width)
        self.vision_encoder = BertModel(visual_bert_config) 

        bert_config = BertConfig.from_pretrained(f'{args.bert_path}/config.json')
        self.roberta = BertModel.from_pretrained(args.bert_path)
    
        text_width = self.roberta.config.hidden_size

        cross_bert_config = BertConfig.from_pretrained(f'{args.bert_path}/config.json')
        self.cross_bert = CrossAttentionEncoder(cross_bert_config, args.cross_num_layers)
        
        self.cls = BertOnlyMLMHead(bert_config)
        self.lm = MaskLM(tokenizer_path=args.bert_path, mlm_probability=0.15)
        self.vocab_size = bert_config.vocab_size
        

    def forward(self, inputs, task=None, inference=False):
        if self.clip_pretrained_path is not None:
            frames = inputs['frame_input']
            B, N, C, H, W = frames.shape
            output_shape = (B, N, -1)
            frames = frames.view(B * N, C, H, W)
            video_inputs = self.visual_backbone(frames)['pooler_output'].view(*output_shape)
        else:
            video_inputs = inputs['frame_input']
        video_mask = inputs['frame_mask']
        text_input_ids = inputs['title_input']
        text_mask = inputs['title_mask']

        # 添加CLS
        B,_ = video_mask.shape
        video_cls_tokens = self.cls_token.expand(B, -1, -1).to(video_inputs.device)
        video_inputs = torch.cat((video_cls_tokens, video_inputs), dim=1)
        video_cls_mask = torch.ones(B,1).to(video_mask.device)
        video_mask = torch.cat((video_cls_mask, video_mask), 1)
        video_output = self.vision_encoder(inputs_embeds=video_inputs, attention_mask=video_mask, output_hidden_states=True)
        video_embeds = video_output[0]
        
        input_ids, _, lm_label = self.lm.torch_mask_tokens(text_input_ids.cpu(),text_mask.cpu())
        text_input_ids = input_ids.to(text_input_ids.device)
        lm_label = lm_label[:, 1:].to(text_input_ids.device) # [SEP] 卡 MASK 大师 [SEP]
        text_output = self.roberta(text_input_ids, attention_mask = text_mask)   
        text_embeds = text_output[0]

        lang_attention_mask = text_mask[:, None, None, :]
        lang_attention_mask = (1.0 - lang_attention_mask) * -10000.0
        visual_attention_mask = video_mask[:, None, None, :]
        visual_attention_mask = (1.0 - visual_attention_mask) * -10000.0
        vision_hidden_states, language_hidden_states = self.cross_bert(lang_feats=text_embeds,
                                                                        lang_attention_mask=lang_attention_mask, 
                                                                        visual_feats=video_embeds, 
                                                                        visual_attention_mask=visual_attention_mask)
        encoder_outputs = language_hidden_states[-1]
        lm_prediction_scores = self.cls(encoder_outputs)[:, 1:text_input_ids.size()[1] , :]
        pred = lm_prediction_scores.contiguous().view(-1, self.vocab_size)
        loss_mlm = nn.CrossEntropyLoss()(pred, lm_label.contiguous().view(-1))

        return loss_mlm
    
class CrossAttentionEncoder(nn.Module):
    def __init__(self, config, num_layers=2):
        super().__init__()
        self.x_layers = nn.ModuleList([LxmertXLayer(config) for _ in range(num_layers)])
    
    def forward(self, lang_feats, lang_attention_mask, visual_feats, visual_attention_mask, output_attentions=None):
        vision_hidden_states = ()
        language_hidden_states = ()
        for layer_module in self.x_layers:
            x_outputs = layer_module(
                lang_feats,
                lang_attention_mask,
                visual_feats,
                visual_attention_mask,
                output_attentions=output_attentions,
            )
            lang_feats, visual_feats = x_outputs[:2]
            vision_hidden_states = vision_hidden_states + (visual_feats,)
            language_hidden_states = language_hidden_states + (lang_feats,)
        return vision_hidden_states, language_hidden_states