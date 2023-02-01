import sys
sys.path.append(".")

import torch
from torch import  nn
from torch.nn import MultiheadAttention
from torch.nn import BCELoss

from transformers import BertPreTrainedModel,BertModel
from transformers.modeling_outputs import SequenceClassifierOutput



def focal_loss_with_prob(pred_logits,
                            labels,
                            gamma=2.0,
                            alpha=0.25,
                            reduction='mean',
                            ):
    """
    PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.
    Different from `py_sigmoid_focal_loss`, this function accepts probability
    as input.

    # https://zhuanlan.zhihu.com/p/80594704
    
    Args:
        pred_logits (torch.Tensor): The prediction probability with shape (N, C),
            C is the number of classes.
        labels (torch.Tensor): The learning label of the prediction.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
            the loss. Defaults to None.
    """

    labels = labels.type_as(pred_logits)
    gamma_base = (1 - pred_logits) * labels + pred_logits * (1 - labels)
    focal_weight = (
        alpha * labels + 
        (1 - alpha) *(1 - labels)
                    ) * gamma_base.pow(gamma)
    
    loss_fct_1 = BCELoss(weight = focal_weight) 
    # BCELoss() 要先sigmoid再传入
    loss = loss_fct_1(pred_logits, labels,reduction = reduction) 
    
    return loss


class BaseMultiLabelClf(BertPreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.config.problem_type = "multi_label_classification"

        self.bert = BertModel(config)

        classifier_dropout = config.hidden_dropout_prob
        
        self.dropout = nn.Dropout(classifier_dropout)
        self.fc = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

        # 初始化混淆矩阵中各个分类情况的权重
        self.true_positive = 1# 预计为真，实际上也为真
        self.false_positive = 1 # 预计为真，但实际上为假的
        self.false_negative =  1# 预计为假，但实际上为真
        self.true_negative =1 # 预计为假，实际上也为假    
        self.loss_type = "bce"

    def init_label_weight(self):
        self.label_weight = nn.Parameter(torch.ones(self.config.num_labels))
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size = input_ids.size(0)
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1] # B * H


        fc_out = self.fc(pooled_output)

        logits = torch.sigmoid(fc_out)


        loss = None

        if labels is not None:
            # 使用 BCELoss
            if self.loss_type == "bce":
                loss_fct_1 = BCELoss() 
                # BCELoss() 要先sigmoid再传入
                loss = loss_fct_1(logits, labels.float()) 
            # 使用
            elif self.loss_type == "focal":
                loss = focal_loss_with_prob(logits, labels.float())
            else:
                exit()

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

