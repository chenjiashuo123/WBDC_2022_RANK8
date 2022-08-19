import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.category_id_map import CATEGORY_ID_LIST
from transformers import CLIPVisionModel, CLIPVisionConfig
from transformers.models.bert.modeling_bert import BertConfig, BertOnlyMLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder, BertPooler
class MULTIBERT(nn.Module):
    def __init__(self, args, task=['tag'], init_from_pretrain=True):
        super(MULTIBERT, self).__init__()
        uni_bert_cfg = BertConfig.from_pretrained(f'{args.bert_path}/config.json')
        self.newfc_hidden = torch.nn.Linear(args.frame_embedding_size, args.vlad_hidden_size)
        if args.init_clip_from_pretrain:
            clip_cfg = CLIPVisionConfig.from_pretrained(f'{args.clip_pretrained_path}/config.json')
            self.visual_backbone = CLIPVisionModel(clip_cfg)
            checkpoint = torch.load(args.resume_checkpoint)
            self.visual_backbone.load_state_dict(checkpoint['model_image'])
            print(f'resume pretrain weight from {args.resume_checkpoint}')
            del checkpoint
        else:
            print(f'clip weight load from {args.clip_pretrained_path}')
            self.visual_backbone = CLIPVisionModel.from_pretrained(args.clip_pretrained_path)
        self.max_frames = args.max_frames
        self.task = set(task)
        self.loss = args.loss
        self.mask_word = args.mask_word
        self.cls_layers = args.cls_layers
        self.newfc_tag =  nn.Linear(uni_bert_cfg.hidden_size, len(CATEGORY_ID_LIST))
        self.roberta = UniBertForMaskedLM.from_pretrained(f'{args.bert_path}')
    def forward(self, inputs, task=None, inference=False):
        frames = inputs['frame_input']
        B, N, C, H, W = frames.shape
        output_shape = (B, N, -1)
        frames = frames.view(B * N, C, H, W)
        video_feature = self.visual_backbone(frames)['pooler_output'].view(*output_shape)
        
        video_mask = inputs['frame_mask']
        text_input_ids = inputs['title_input']
        text_mask = inputs['title_mask']
        features, hidden_states, lm_prediction_scores = self.roberta(video_feature, video_mask, text_input_ids, text_mask, return_mlm=False)
        embedding = torch.mean(features, 1)
        prediction = self.newfc_tag(embedding)
        if inference:
            return torch.argmax(prediction, dim=1), prediction
        else:
            loss, accuracy, pred_label_id, label =  self.calculate_tag_loss(prediction, inputs['label'])
            return loss, accuracy, pred_label_id, label
    def calculate_tag_loss(self, prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label
def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}
class VisualPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class VisualLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = VisualPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, 768, bias=False)
        self.bias = nn.Parameter(torch.zeros(768))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class VisualOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = VisualLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores
    
class UniBertForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = UniBert(config)
        self.cls = BertOnlyMLMHead(config)
    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(self, video_feature, video_mask, text_input_ids, text_mask, gather_index=None, return_mlm=False):
        encoder_outputs, hidden_states = self.bert(video_feature, video_mask, text_input_ids, text_mask)
        if return_mlm:
            return encoder_outputs, hidden_states,self.cls(encoder_outputs)[:, 1:text_input_ids.size()[1] , :]
        else:
            return encoder_outputs, hidden_states, None        
        
class UniBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.video_fc = torch.nn.Linear(768, config.hidden_size)
        self.video_embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(self, video_feature, video_mask, text_input_ids, text_mask, gather_index=None):        
        text_emb = self.embeddings(input_ids=text_input_ids)
        # reduce frame feature dimensions : 1536 -> 1024
        video_emb = self.video_fc(video_feature)
        video_emb = self.video_embeddings(inputs_embeds=video_emb)
        mask = torch.cat([text_mask, video_mask], dim=-1)
        embedding_output = torch.cat([text_emb, video_emb], dim=1)
        mask = mask[:, None, None, :]
        mask = (1.0 - mask) * -10000.0
        # mask = mask.half()
        outputs = self.encoder(embedding_output, attention_mask=mask, output_hidden_states=True)
        encoder_outputs = outputs['last_hidden_state']
        hidden_states = outputs['hidden_states']
        pooled_output = self.pooler(encoder_outputs)
        return encoder_outputs, hidden_states