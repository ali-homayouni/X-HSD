import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def save_image(epoch):
    dirname = './img/cm/'
    if not os.path.exists(dirname):
        os.makedirs(dirname)    
    plt.savefig(dirname + str(epoch) + '-all' +'.png')

def save(toBeSaved, filename, mode='wb'):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(filename, mode) as file:
        pickle.dump(toBeSaved, file)

def save_hugging_face(model, dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    model.save(dirname)

def save_tokenizer(tokenizer, dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    tokenizer.save_pretrained(dirname)

def load(filename, mode='rb'):
    with open(filename, mode) as file:
        loaded = pickle.load(file)
    return loaded

def pad_sents(sents, pad_token):
    sents_padded = []
    lens = get_lens(sents)
    max_len = max(lens)
    sents_padded = [sents[i] + [pad_token] * (max_len - l) for i, l in enumerate(lens)]
    return sents_padded

def sort_sents(sents, reverse=True):
    sents.sort(key=(lambda s: len(s)), reverse=reverse)
    return sents

def get_mask(sents, unmask_idx=1, mask_idx=0):
    lens = get_lens(sents)
    max_len = max(lens)
    mask = [([unmask_idx] * l + [mask_idx] * (max_len - l)) for l in lens]
    return mask

def get_lens(sents):
    return [len(sent) for sent in sents]

def get_max_len(sents):
    max_len = max([len(sent) for sent in sents])
    return max_len

def truncate_sents(sents, length):
    sents = [sent[:length] for sent in sents]
    return sents

def get_loss_weight(labels, label_order):
    nums = [np.sum(labels == lo) for lo in label_order]
    loss_weight = torch.tensor([n / len(labels) for n in nums])
    return loss_weight
