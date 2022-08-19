import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.category_id_map import CATEGORY_ID_LIST
from dataset.pretrain_helper import MaskLM, MaskVideo, ShuffleVideo, MMM_Mask
from transformers.models.bert.modeling_bert import BertConfig, BertOnlyMLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder, BertPooler
class MULTIBERT(nn.Module):
    def __init__(self, args, task=['tag'], init_from_pretrain=True):
        super(MULTIBERT, self).__init__()
        uni_bert_cfg = BertConfig.from_pretrained(f'{args.bert_path}/config.json')
        self.newfc_hidden = torch.nn.Linear(args.frame_embedding_size, args.vlad_hidden_size)
        self.max_frames = args.max_frames
        self.task = set(task)
        self.loss = args.loss
        self.mask_word = args.mask_word
        if 'tag' in task:
            self.newfc_tag =  nn.Linear(uni_bert_cfg.hidden_size, len(CATEGORY_ID_LIST))
        if 'mlm' in task:
            self.lm = MaskLM(tokenizer_path=args.bert_path, mlm_probability=0.3)
            self.vocab_size = uni_bert_cfg.vocab_size
        if 'itm' in task:
            self.sv = ShuffleVideo()
            self.newfc_itm = torch.nn.Linear(uni_bert_cfg.hidden_size, 2)
        if 'mmm' in task:
            self.mmm_mask = MMM_Mask(tokenizer_path=args.bert_path)
            self.vocab_size = uni_bert_cfg.vocab_size
        if 'mmm' in task or 'mfm' in task:
            self.roberta_mvm_lm_header = VisualOnlyMLMHead(uni_bert_cfg)
        if 'mfm' in task:
            self.mfm_vm = MaskVideo(mlm_probability=0.15)
        if init_from_pretrain:
            self.roberta = UniBertForMaskedLM.from_pretrained(f'{args.bert_path}')
        else:
            self.roberta = UniBertForMaskedLM(uni_bert_cfg)
    def forward(self, inputs, task=None, inference=False):
        video_feature = inputs['frame_input']
        video_mask = inputs['frame_mask']
        text_input_ids = inputs['title_input']
        text_mask = inputs['title_mask']
        bs = video_mask.shape[0]
        mmm_bs = bs // 4
        if task is None:
            sample_task = self.task
        elif type(task) == str:
            sample_task = [task]
        elif type(task) == list:
            sample_task = task
        # compute fintune task loss
        return_mlm = False
        mmm_video_feature = video_feature[0:mmm_bs, :, :]
        mmm_video_mask = video_mask[0:mmm_bs, :]
        mmm_text_input_ids = text_input_ids[0:mmm_bs, :]
        mmm_text_mask = text_mask[0:mmm_bs, :]

        left_video_feature = video_feature[mmm_bs:, :, :]
        left_video_mask = video_mask[mmm_bs:, :]
        left_text_input_ids = text_input_ids[mmm_bs:, :]
        left_text_mask = text_mask[mmm_bs:, :]

        
        if 'mmm' in sample_task:
            mmm_input_feature, mmm_video_label = self.mmm_mask.torch_mask_frames(mmm_video_feature.cpu(), mmm_video_mask.cpu())
            mmm_video_feature = mmm_input_feature.to(video_feature.device)
            mmm_video_label = mmm_video_label.to(video_feature.device)
            mmm_input_ids, mmm_t_label = self.mmm_mask.torch_mask_tokens(mmm_text_input_ids.cpu(),mmm_text_mask.cpu())
            mmm_text_input_ids = mmm_input_ids.to(text_input_ids.device)
            mmm_t_label = mmm_t_label.to(text_input_ids.device)
            return_mlm = True

        if 'mlm' in sample_task:
            mlm_input_ids, _, lm_label = self.lm.torch_mask_tokens(left_text_input_ids.cpu(),left_text_mask.cpu())
            left_text_input_ids = mlm_input_ids.to(left_text_input_ids.device)
            mlm_lm_label = lm_label.to(left_text_input_ids.device)
            return_mlm = True

        if 'mfm' in sample_task:
            mfm_input_feature, mfm_video_label = self.mfm_vm.torch_mask_frames(left_video_feature.cpu(), left_video_mask.cpu())
            left_video_feature = mfm_input_feature.to(video_feature.device)
            mfm_video_label = mfm_video_label.to(video_feature.device)

        if 'itm' in sample_task:
            input_feature, input_mask, video_text_match_label = self.sv.torch_shuf_video(left_video_feature.cpu(), left_video_mask.cpu())
            left_video_feature = input_feature.to(video_feature.device)
            left_video_mask = input_mask.to(video_mask.device)
            video_text_match_label = video_text_match_label.to(video_feature.device, dtype=torch.long)

        video_feature = torch.cat((mmm_video_feature, left_video_feature), 0)
        video_mask = torch.cat((mmm_video_mask, left_video_mask), 0)
        text_input_ids = torch.cat((mmm_text_input_ids, left_text_input_ids), 0)
        text_mask = torch.cat((mmm_text_mask, left_text_mask), 0)
        features, pooled_output, lm_prediction_scores = self.roberta(video_feature, video_mask, text_input_ids, text_mask, return_mlm=return_mlm)
        loss = 0
        if 'mlm' in sample_task:
            mlm_prediction_scores = lm_prediction_scores[mmm_bs:,:,:]
            pred = mlm_prediction_scores.contiguous().view(-1, self.vocab_size)
            masked_lm_loss = nn.CrossEntropyLoss()(pred, mlm_lm_label.contiguous().view(-1))
            loss += masked_lm_loss
        if 'itm' in sample_task:
            pred = self.newfc_itm(features[mmm_bs:, 0, :])
            itm_loss = nn.CrossEntropyLoss()(pred.view(-1, 2), video_text_match_label.contiguous().view(-1))
            loss += itm_loss
        if 'mfm' in sample_task:
            vm_output = self.roberta_mvm_lm_header(features[mmm_bs:, text_input_ids.size()[1]:, :])
            masked_vm_loss = self.calculate_mfm_loss(vm_output, inputs['frame_input'][mmm_bs:, :, :], 
                                                     left_video_mask, mfm_video_label, normalize=False)
            masked_vm_loss = masked_vm_loss / 2
            loss += masked_vm_loss
        if 'mmm' in sample_task:
            mmm_text_bs = mmm_bs//2
            mmm_prediction_scores = lm_prediction_scores[mmm_text_bs:mmm_bs,:,:]
            mmm_pred = mmm_prediction_scores.contiguous().view(-1, self.vocab_size)
            cap_l_loss = nn.CrossEntropyLoss()(mmm_pred, mmm_t_label.contiguous().view(-1))
            loss += cap_l_loss
            cap_l_loss = cap_l_loss / 2
            mmm_vm_output = self.roberta_mvm_lm_header(features[:mmm_text_bs, text_input_ids.size()[1]:, :])
            cap_v_loss = self.calculate_mfm_loss(mmm_vm_output, inputs['frame_input'][:mmm_text_bs, :, :], 
                                                     mmm_video_mask[:mmm_text_bs, :], mmm_video_label, normalize=False)
            cap_v_loss = cap_v_loss / 2
            loss += cap_v_loss
        if 'ict' in sample_task:
            ict_title_input_ids = inputs['title_input']
            ict_title_mask = inputs['title_mask']
            asr_ocr_input_ids = inputs['asr_ocr_input_ids']
            asr_ocr_mask = inputs['asr_ocr_mask']
            ict_video_mask = inputs['frame_mask']
            ict_video_feature = inputs['frame_input']

            title_feature, _, _ = self.roberta(ict_video_feature, ict_video_mask, ict_title_input_ids, ict_title_mask, text_only=True,return_mlm=False)
            ict_features, _, _ = self.roberta(ict_video_feature, ict_video_mask, asr_ocr_input_ids, asr_ocr_mask, text_only=False,return_mlm=False)
            ict_mask = torch.cat([asr_ocr_mask, ict_video_mask], dim=-1)
            ict_emb = (ict_features * ict_mask.unsqueeze(-1)).sum(1)/(ict_mask.sum(-1)+1e-10).unsqueeze(-1)
            title_emb = (title_feature * ict_title_mask.unsqueeze(-1)).sum(1)/(ict_title_mask.sum(-1)+1e-10).unsqueeze(-1)

            ict_emb = F.normalize(ict_emb, dim=-1)
            title_emb = F.normalize(title_emb, dim=-1)
            sim_v2t = ict_emb @ title_emb.t()
            sim_t2v = sim_v2t.t()
            sim_targets = torch.zeros(sim_v2t.size()).to(ict_emb.device)
            sim_targets.fill_diagonal_(1)
            loss_v2t = -torch.sum(F.log_softmax(sim_v2t, dim=1)*sim_targets,dim=1).mean()
            loss_t2v = -torch.sum(F.log_softmax(sim_t2v, dim=1)*sim_targets,dim=1).mean()
            loss_ict = loss_v2t+loss_t2v
            loss += loss_ict
        embedding = torch.mean(features, 1)
        if 'tag' in sample_task:
            prediction = self.newfc_tag(embedding)
            if inference:
                return torch.argmax(prediction, dim=1), prediction
            else:
                tag_loss, accuracy, pred_label_id, label =  self.calculate_tag_loss(prediction, inputs['label'])
                loss += tag_loss*3
                return loss, accuracy, pred_label_id, label
        return loss, masked_lm_loss, itm_loss, loss_ict, masked_vm_loss, cap_l_loss, cap_v_loss
    def calculate_tag_loss(self, prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label

    def calculate_mfm_loss(self, video_feature_output, video_feature_input, 
                           video_mask, video_labels_index, normalize=False, temp=0.1):
        if normalize:
            video_feature_output = torch.nn.functional.normalize(video_feature_output, p=2, dim=2)
            video_feature_input = torch.nn.functional.normalize(video_feature_input, p=2, dim=2)

        afm_scores_tr = video_feature_output.view(-1, video_feature_output.shape[-1])

        video_tr = video_feature_input.permute(2, 0, 1)
        video_tr = video_tr.view(video_tr.shape[0], -1)

        logits_matrix = torch.mm(afm_scores_tr, video_tr)
        if normalize:
            logits_matrix = logits_matrix / temp

        video_mask_float = video_mask.to(dtype=torch.float)
        mask_matrix = torch.mm(video_mask_float.view(-1, 1), video_mask_float.view(1, -1))
        masked_logits = logits_matrix + (1. - mask_matrix) * -1e8

        logpt = F.log_softmax(masked_logits, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt

        video_labels_index_mask = (video_labels_index != -100)
        nce_loss = nce_loss.masked_select(video_labels_index_mask.view(-1))
        nce_loss = nce_loss.mean()
        return nce_loss
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
    def forward(self, video_feature, video_mask, text_input_ids, text_mask, text_only=False, gather_index=None, return_mlm=False):
        encoder_outputs, pooled_output = self.bert(video_feature, video_mask, text_input_ids, text_mask, text_only = text_only)
        if return_mlm:
            return encoder_outputs, pooled_output,self.cls(encoder_outputs)[:, :text_input_ids.size()[1] , :]
        else:
            return encoder_outputs, pooled_output, None        
        
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
    def forward(self, video_feature, video_mask, text_input_ids, text_mask, text_only=False,gather_index=None): 
        if text_only:
            embedding_output = self.embeddings(input_ids=text_input_ids)
            mask = text_mask[:, None, None, :]
            mask = (1.0 - mask) * -10000.0
        else:
            text_emb = self.embeddings(input_ids=text_input_ids)
            # reduce frame feature dimensions : 1536 -> 1024
            video_emb = self.video_fc(video_feature)
            video_emb = self.video_embeddings(inputs_embeds=video_emb)
            mask = torch.cat([text_mask, video_mask], dim=-1)
            embedding_output = torch.cat([text_emb, video_emb], dim=1)
            mask = mask[:, None, None, :]
            mask = (1.0 - mask) * -10000.0
        encoder_outputs = self.encoder(embedding_output, attention_mask=mask)['last_hidden_state']
        pooled_output = self.pooler(encoder_outputs)
        return encoder_outputs, pooled_output