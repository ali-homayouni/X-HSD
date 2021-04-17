import torch
from torch import nn
from transformers import (
    BertForSequenceClassification, 
    RobertaForSequenceClassification,
    )


class BERT(nn.Module):
    def __init__(self, model_size, args, num_labels=2):
        super(BERT, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(
            f'bert-{model_size}-uncased',
            num_labels=num_labels,
            hidden_dropout_prob=args['hidden_dropout'],
            attention_probs_dropout_prob=args['attention_dropout']
        )

        # Freeze embeddings' parameters for saving memory
        # for param in self.model.bert.embeddings.parameters():
        #     param.requires_grad = False

    def forward(self, inputs, lens, mask, labels=None):
        outputs = self.model(inputs, attention_mask=mask)
        logits = outputs[0]
        # return loss, logits
        return logits

class RoBERTa(nn.Module):
    def __init__(self, model_size, args, num_labels=2):
        super(RoBERTa, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained(
            f'roberta-{model_size}',
            num_labels=num_labels,
            hidden_dropout_prob=args['hidden_dropout'],
            attention_probs_dropout_prob=args['attention_dropout']
        )

        # Freeze embeddings' parameters for saving memory
        # for param in self.model.roberta.embeddings.parameters():
        #     param.requires_grad = False

    def forward(self, inputs, lens, mask, labels):
        outputs = self.model(inputs, attention_mask=mask, labels=labels)
        loss, logits = outputs[:2]
        # return loss, logits
        return logits

class XLM_RoBERTa(nn.Module):
    def __init__(self, model_size, args, num_labels=2):
        super(XLM_RoBERTa, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained(
            f'xlm-roberta-{model_size}',
            num_labels=num_labels,
            hidden_dropout_prob=args['hidden_dropout'],
            attention_probs_dropout_prob=args['attention_dropout']
        )

        # Freeze embeddings' parameters for saving memory
        for param in self.model.roberta.embeddings.parameters():
            param.requires_grad = False

    def forward(self, inputs, lens, mask, labels):
        outputs = self.model(inputs, attention_mask=mask, labels=labels)
        loss, logits = outputs[:2]
        # return loss, logits
        return logits

