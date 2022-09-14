import torch
from torch import nn
from transformers import (
    BertForSequenceClassification, 
    RobertaForSequenceClassification,
    BertModel,
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
        if args['freeze']:
            for param in self.model.bert.parameters():
                param.requires_grad = False

    def forward(self, inputs, lens, mask, labels=None):
        outputs = self.model(inputs, attention_mask=mask)
        logits = outputs[0]
        return logits
    
    def save(self, filepath):
        self.model.save_pretrained(filepath)

class BERTTWEET_FA(nn.Module):
    def __init__(self, model_size, args, num_labels=2):
        super(BERTTWEET_FA, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(
            f'arm-on/BERTweet-FA',
            num_labels=num_labels,
            hidden_dropout_prob=args['hidden_dropout'],
            attention_probs_dropout_prob=args['attention_dropout']
        )

        # Freeze embeddings' parameters for saving memory
        if args['freeze']:
            for param in self.model.bert.parameters():
                param.requires_grad = False

    def forward(self, inputs, lens, mask):
        outputs = self.model(inputs, attention_mask=mask)
        logits = outputs[0]
        return logits
        
    def save(self, filepath):
        self.model.save_pretrained(filepath)

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
        if args['freeze']:
            for param in self.model.bert.parameters():
                param.requires_grad = False

    def forward(self, inputs, lens, mask):
        outputs = self.model(inputs, attention_mask=mask)
        logits = outputs[0]
        return logits
    
    def save(self, filepath):
        self.model.save_pretrained(filepath)

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
        if args['freeze']:
            for param in self.model.bert.parameters():
                param.requires_grad = False

    def forward(self, inputs, lens, mask):
        outputs = self.model(inputs, attention_mask=mask)
        logits = outputs[0]
        return logits
    
    def save(self, filepath):
        self.model.save_pretrained(filepath)

class ParsBERT(nn.Module):
    def __init__(self, model_size, args, num_labels=2):
        super(ParsBERT, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained(
            f'HooshvareLab/bert-{model_size}-parsbert-uncased',
            num_labels=num_labels,
            hidden_dropout_prob=args['hidden_dropout'],
            attention_probs_dropout_prob=args['attention_dropout']
        )

        # Freeze embeddings' parameters for saving memory
        if args['freeze']:
            for param in self.model.bert.parameters():
                param.requires_grad = False

    def forward(self, inputs, lens, mask):
        outputs = self.model(inputs, attention_mask=mask)
        logits = outputs[0]
        return logits

    def save(self, filepath):
        self.model.save_pretrained(filepath)

class MultilingualBERT(nn.Module):
    def __init__(self, model_size, args, num_labels=2):
        super(MultilingualBERT, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(
            f'bert-{model_size}-multilingual-uncased',
            num_labels=num_labels,
            hidden_dropout_prob=args['hidden_dropout'],
            attention_probs_dropout_prob=args['attention_dropout']
        )

        # Freeze embeddings' parameters for saving memory
        if args['freeze']:
            for param in self.model.bert.parameters():
                param.requires_grad = False

    def forward(self, inputs, lens, mask):
        outputs = self.model(inputs, attention_mask=mask)
        logits = outputs[0]
        return logits

    def save(self, filepath):
        self.model.save_pretrained(filepath)

class GE_BERT(nn.Module):
    def __init__(self, model_size, args, num_labels=2):
        super(GE_BERT, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(
            f'bert-{model_size}-german-dbmdz-uncased',
            num_labels=num_labels,
            hidden_dropout_prob=args['hidden_dropout'],
            attention_probs_dropout_prob=args['attention_dropout']
        )

        # Freeze embeddings' parameters for saving memory
        if args['freeze']:
            for param in self.model.bert.parameters():
                param.requires_grad = False

    def forward(self, inputs, lens, mask):
        outputs = self.model(inputs, attention_mask=mask)
        logits = outputs[0]
        return logits

    def save(self, filepath):
        self.model.save_pretrained(filepath)

class MiniBert(nn.Module):
    def __init__(self, model_size, args, word_embedding, num_labels=2):
        super(MiniBert, self).__init__()
        self.bert = BertModel.from_pretrained(
            f'bert-{model_size}-uncased',
            hidden_dropout_prob=args['hidden_dropout'],
            attention_probs_dropout_prob=args['attention_dropout']
        )
        self.embedding_dim = self.bert.config.hidden_size
        self.fc1 = nn.Linear(self.embedding_dim, word_embedding)
        self.fc2 = nn.Linear(word_embedding, num_labels)

        # Freeze embeddings' parameters for saving memory
        if args['freeze']:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, inputs, lens, mask):
        sequence_output, pooled_output = self.bert(inputs, attention_mask=mask)
        out = self.fc1(sequence_output[:, 0, :])
        out = self.fc2(out)
        return out

    def save(self, filepath):
        self.bert.save_pretrained(filepath)