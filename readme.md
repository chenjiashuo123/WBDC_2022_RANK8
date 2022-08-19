# WBDC_2022_RANK8
2022微信大数据挑战赛 第8名 方案

官网链接 : [link](https://algo.weixin.qq.com/) <br> 

## 环境配置
Python 版本：3.8<br>
PyTorch 版本：1.9.0<br>
CUDA 版本：11.1<br>

所需环境在 `requirements.txt` 中定义。

## 数据
* 使用大赛提供的未标注数据进行无监督预训练（100万）
* 使用大赛提供的有标注数据进行微调（10万）。

## 简介
### 模型简介
(1) 单流模型: visual-bert [郭大分享链接](https://developers.weixin.qq.com/community/minihome/article/doc/000e22f25e0cc026a4fd55bce51013)

(2) 双流模型：lxmert
* 文本：bert-base
* 视频：vit-base
* 融合层：三层cross attention

### 视频特征抽取
* 使用了 huggingface 上提供的 `openai/clip-vit-base-patch32` 模型。[link](https://huggingface.co/openai/clip-vit-base-patch32)


### 单流预训练
(1) Mask language model 任务<br> 
* title 随机 30% 进行 mask，预测 mask 词。<br>

(2) Mask frame model 任务<br> 
* 对 frame 的随机 15% 进行 mask，mask 采用了全 0 的向量填充。<br>
采用了 Nce loss，最大化 mask 帧和预测帧的互信息<br>

(3) frame text match 任务<br> 
* 打乱 batch 中 50%的视频帧， 预测 frame 和 text 是否配对<br>

(4) frame text clip 任务<br>
* 参考 郭大QQ浏览的 Inverse Cloze Task 的做法，[link](https://pcg-kandian-alg-race-1251316161.cos.ap-guangzhou.myqcloud.com/ws/topic1/top2/%E8%8B%9F%E8%BF%9B%E5%86%B3%E8%B5%9B.pdf) <br>
* 具体实现：<br> 
  * title 单独输入bert 得到 title_embedding<br>
  * frame ocr asr 拼接后 输入 bert 得到 frame_ocr_asr_embedding<br>
  * 构建 title_embedding 和 frame_ocr_asr_embedding 对比loss<br>

(5) mask modality clip 任务<br>
  * 针对模态缺失的问题：设计 mask 掉 全部 frame 或 全部 text 的 任务<br>
  * 参考论文： VLM: Task-agnostic Video-Language Model Pre-training for Video Understanding [link](https://arxiv.org/abs/2105.09996)<br>

### clip预训练
(1) video text clip 任务<br>
* 参照论文：ActionCLIP: A New Paradigm for Video Action Recognition [link](https://arxiv.org/abs/2109.08472)
* 具体实现
  * 使用 clip-vit-base 提取 每个视频4帧的图像特征后 mean pooling 代表 video_embedding
  * 使用 bert 提取 tilte 的特征 代表 title_embedding
  *  video_embedding 和 title_embedding 做 clip loss

### 微调
| 模型 id | 模型结构 | bert 初始化权重 | vit 初始化权重 | trick | F1-mean |
| ------- | ------- | -------------- | ------------- | ----- | ------- |
| model-1 | 单流 | 单流预训练 epoch 15 | clip-vit-base-32 | ema fgm | 71.4 (单折)<br>72.2 (全量) |
| model-2 | 单流 | 单流预训练 epoch 15 | clip 预训练 epoch 15 | ema fgm | 71.7(单折) 72.1(全量) |
| model-3 | 双流 | macbert-base | clip-vit-base-32 | ema fgm | 71(单折) |
| model-4 | 单流 | 单流预训练 epoch 15 | clip-vit-base-32 | ema pgd | - |

4个模型 ensemble 复赛 F1-mean：0.731633

## 最后
* 历时两个半月，非常感谢主办方提供的数据和计算资源，感谢工作人员的辛苦答疑。
* 此外， 也感谢QQ浏览器Ai算法大赛第一名、第二名和初赛周周星各位大佬们的分享和无私开源，我从中学到了许多无监督预训和训练技巧的新知识。








