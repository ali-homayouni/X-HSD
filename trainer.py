# Built-in libraries
import copy
import datetime
from typing import Dict, List
# Third-party libraries
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
# Local files
from utils import save
from config import LABEL_DICT_OLID

class Trainer():
    '''
    The trainer for training models.
    It can be used for both single and multi task training.
    Every class function ends with _m is for multi-task training.
    '''
    def __init__(
        self,
        model: nn.Module,
        epochs: int,
        dataloaders: Dict[str, DataLoader],
        criterion: nn.Module,
        loss_weights: List[float],
        clip: bool,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        device: str,
        patience: int,
        task_name: str,
        model_name: str,
        seed: int
    ):
        self.model = model
        self.epochs = epochs
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.loss_weights = loss_weights
        self.clip = clip
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.patience = patience
        self.task_name = task_name
        self.model_name = model_name
        self.seed = seed
        self.datetimestr = datetime.datetime.now().strftime('%Y-%b-%d_%H:%M:%S')

        # Evaluation results
        self.train_losses = []
        self.test_losses = []
        self.train_f1 = []
        self.test_f1 = []
        self.best_train_f1 = 0.0
        self.best_test_f1 = 0.0

        # Evaluation results for multi-task
        self.best_train_f1_m = np.array([0, 0, 0], dtype=np.float64)
        self.best_test_f1_m = np.array([0, 0, 0], dtype=np.float64)

    def train(self):
        for epoch in range(1, self.epochs+1):
            print(f'Epoch {epoch}')
            print('=' * 20)
            self.train_one_epoch()
            self.test()
            print(f'Best test f1: {self.best_test_f1:.4f}')
            print('=' * 20)

        print('Saving results ...')
        save(
            (self.train_losses, self.test_losses, self.train_f1, self.test_f1, self.best_train_f1, self.best_test_f1),
            f'./save/results/single_{self.task_name}_{self.datetimestr}_{self.best_test_f1:.4f}.pt'
        )

    def train_one_epoch(self):
        self.model.train()
        dataloader = self.dataloaders['train']
        y_pred_all = None
        labels_all = None
        loss = 0
        iters_per_epoch = 0
        for inputs, lens, mask, labels in tqdm(dataloader, desc='Training'):
            iters_per_epoch += 1

            if labels_all is None:
                labels_all = labels.numpy()
            else:
                labels_all = np.concatenate((labels_all, labels.numpy()))

            inputs = inputs.to(device=self.device)
            lens = lens.to(device=self.device)
            mask = mask.to(device=self.device)
            labels = labels.to(device=self.device)

            self.optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # Forward
                logits = self.model(inputs, lens, mask, labels)
                _loss = self.criterion(logits, labels)
                loss += _loss.item()
                y_pred = logits.argmax(dim=1).cpu().numpy()

                if y_pred_all is None:
                    y_pred_all = y_pred
                else:
                    y_pred_all = np.concatenate((y_pred_all, y_pred))

                # Backward
                _loss.backward()
                if self.clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

        loss /= iters_per_epoch
        f1 = f1_score(labels_all, y_pred_all, average='macro')

        print(f'loss = {loss:.4f}')
        print(f'Macro-F1 = {f1:.4f}')

        self.train_losses.append(loss)
        self.train_f1.append(f1)
        if f1 > self.best_train_f1:
            self.best_train_f1 = f1

    def test(self):
        self.model.eval()
        dataloader = self.dataloaders['test']
        y_pred_all = None
        labels_all = None
        loss = 0
        iters_per_epoch = 0
        for inputs, lens, mask, labels in tqdm(dataloader, desc='Testing'):
            iters_per_epoch += 1

            if labels_all is None:
                labels_all = labels.numpy()
            else:
                labels_all = np.concatenate((labels_all, labels.numpy()))

            inputs = inputs.to(device=self.device)
            lens = lens.to(device=self.device)
            mask = mask.to(device=self.device)
            labels = labels.to(device=self.device)

            with torch.set_grad_enabled(False):
                logits = self.model(inputs, lens, mask, labels)
                _loss = self.criterion(logits, labels)
                y_pred = logits.argmax(dim=1).cpu().numpy()
                loss += _loss.item()

                if y_pred_all is None:
                    y_pred_all = y_pred
                else:
                    y_pred_all = np.concatenate((y_pred_all, y_pred))

        loss /= iters_per_epoch
        f1 = f1_score(labels_all, y_pred_all, average='macro')

        print(f'loss = {loss:.4f}')
        print(f'Macro-F1 = {f1:.4f}')

        self.test_losses.append(loss)
        self.test_f1.append(f1)
        if f1 > self.best_test_f1:
            self.best_test_f1 = f1
            self.save_model()

    def calc_f1(self, labels, y_pred):
        return np.array([
            f1_score(labels.cpu(), y_pred.cpu(), average='macro'),
            f1_score(labels.cpu(), y_pred.cpu(), average='micro'),
            f1_score(labels.cpu(), y_pred.cpu(), average='weighted')
        ], np.float64)

    def save_model(self):
        print('Saving model...')
        if self.task_name == 'all':
            filename = f'./save/models/{self.task_name}_{self.model_name}_{self.best_test_f1_m[0]}_seed{self.seed}.pt'
        else:
            filename = f'./save/models/{self.task_name}_{self.model_name}_{self.best_test_f1}_seed{self.seed}.pt'
        save(copy.deepcopy(self.model.state_dict()), filename)
