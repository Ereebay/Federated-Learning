import copy
import os

import torch
import torch.nn as nn
from tqdm import tqdm

import model as fedmodel
from config import cfg

device = "cuda" if torch.cuda.is_available() else "cpu"

KL_Loss = nn.KLDivLoss(reduction='batchmean')
Softmax = nn.Softmax(dim=1)
LogSoftmax = nn.LogSoftmax(dim=1)
CE_Loss = nn.CrossEntropyLoss()


def subtract(target, minuend, subtrahend, agreement):
    for name in target:
        target[name].data = agreement * (minuend[name].data.clone() - subtrahend[name].data.clone())


def reduce_add_average(target, sources):
    for name in target:
        tmp = torch.sum(torch.stack([source[name].data for source in sources]), dim=0).clone()
        target[name].data += tmp


def copy(target, source):
    for name in target:
        target[name].data = source[name].data.clone()


def train_mutual(node_idx, model, localmodel, loader, optimizer, localoptimizer, epochs=1):
    model.train()
    localmodel.train()
    for epoch in range(epochs):
        total_local_loss = 0.0
        avg_local_loss = 0.0
        correct_local = 0.0
        acc_local = 0.0
        total_model_loss = 0.0
        avg_model_loss = 0.0
        correct_model = 0.0
        acc_model = 0.0
        with tqdm(loader) as loop:
            for idx, (data, target) in enumerate(loop):
                optimizer.zero_grad()
                localoptimizer.zero_grad()
                loop.set_description(f'Node{node_idx}: loss_local={avg_local_loss:.4f} acc_local={acc_local:.2f}% \
        loss_model={avg_model_loss:.4f} acc_model={acc_model:.2f}%')
                data, target = data.to(device), target.to(device)
                output_local = localmodel(data)
                output_model = model(data)
                ce_local = CE_Loss(output_local, target)
                kl_local = KL_Loss(LogSoftmax(output_local), Softmax(output_model.detach()))
                ce_model = CE_Loss(output_model, target)
                kl_model = KL_Loss(LogSoftmax(output_model), Softmax(output_local.detach()))
                loss_local = cfg.Optima.alpha * ce_local + (1 - cfg.Optima.alpha) * kl_local
                loss_model = cfg.Optima.beta * ce_model + (1 - cfg.Optima.beta) * kl_model
                loss_local.backward()
                loss_model.backward()
                optimizer.step()
                localoptimizer.step()
                total_local_loss += loss_local
                avg_local_loss = total_local_loss / (idx + 1)
                pred_local = output_local.argmax(dim=1)
                correct_local += pred_local.eq(target.view_as(pred_local)).sum()
                acc_local = correct_local.float() / len(loader.dataset) * 100
                total_model_loss += loss_model
                avg_model_loss = total_model_loss / (idx + 1)
                pred_model = output_model.argmax(dim=1)
                correct_model += pred_model.eq(target.view_as(pred_model)).sum()
                acc_model = correct_model.float() / len(loader.dataset) * 100
    return avg_local_loss, avg_model_loss, acc_local, acc_model


def train_op(node_idx, model, loader, optimizer, epochs=1):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        avg_loss = 0.0
        correct = 0.0
        acc = 0.0
        total_acc = 0.0
        avg_acc = 0.0
        description = "Node{:d}: loss={:.4f} acc={:.2f}%"
        with tqdm(loader) as loop:
            for idx, (data, target) in enumerate(loop):
                optimizer.zero_grad()
                loop.set_description(description.format(node_idx, avg_loss, acc))
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = torch.nn.CrossEntropyLoss()(output, target)
                total_loss += loss
                avg_loss = total_loss / (idx + 1)
                pred = output.argmax(dim=1)
                correct += pred.eq(target.view_as(pred)).sum()
                acc = correct.float() * 100 / len(loader.dataset)
                total_acc += acc
                avg_acc = total_acc / (idx + 1)
                loss.backward()
                optimizer.step()
    return avg_loss, avg_acc


def init_model(model_type):
    model = []
    if model_type == 'LeNet5':
        model = fedmodel.LeNet5()
    elif model_type == 'MLP':
        model = fedmodel.MLP()
    elif model_type == 'ResNet18':
        model = fedmodel.ResNet18()
    elif model_type == 'CNN':
        model = fedmodel.CNN()
    return model


def init_optimizer(optimizer_type, model, learning_rate, momentum):
    optimizer = []
    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=5e-4)
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    return optimizer


def weights_zero(model):
    for p in model.parameters():
        if p.data is not None:
            p.data.detach_()
            p.data.zero_()


def eval_op(model, loader):
    model.eval()
    samples, correct = 0, 0

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            y_ = model(x)
            _, predicted = torch.max(y_.data, 1)

            samples += y.shape[0]
            correct += (predicted == y).sum().item()

    return {"accuracy": correct / samples}


class Node(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.model = init_model(cfg.Model.global_model).to(device)
        self.weight = {key: value for key, value in self.model.named_parameters()}
        self.device = device

    def evaluate(self, loader=None):
        return eval_op(self.model, self.dataloader if not loader else loader)


class Client(Node):
    def __init__(self, idx, dataloader, validloader):
        super().__init__(dataloader)
        self.idx = idx + 1
        self.train_dataloader = dataloader
        self.valid_dataloader = validloader
        self.optimizer = init_optimizer(cfg.Optima.optimizer, self.model, cfg.Optima.meme_lr, cfg.Optima.momentum)
        self.weight_old = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.gradient = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}
        if cfg.Total.alogrithm == 'fed_mutual':
            self.local_model = init_model(cfg.Model.local_model).to(device)
            self.local_optimizer = init_optimizer(cfg.Optima.optimizer, self.local_model, cfg.Optima.local_lr,
                                                  cfg.Optima.momentum)

    def fork(self, global_node):
        copy(target=self.weight, source=global_node.weight)

    def compute_gradients(self, epochs, agreement, loader=None):
        copy(target=self.weight_old, source=self.weight)
        if cfg.Total.alogrithm == 'fed_mutual':
            train_local_loss, train_model_loss, train_local_acc, train_model_acc = train_mutual(self.idx, self.model,
                                                                                                self.local_model,
                                                                                                self.train_dataloader,
                                                                                                self.optimizer,
                                                                                                self.local_optimizer,
                                                                                                epochs)
            metric = (train_local_loss, train_model_loss, train_local_acc, train_model_acc)
        else:
            train_loss, train_acc = train_op(self.idx, self.model, self.train_dataloader, self.optimizer, epochs)
            metric = (train_loss, train_acc)
        subtract(target=self.gradient, minuend=self.weight, subtrahend=self.weight_old, agreement=agreement)

        return metric

    def adapt(self, epochs, loader=None):
        train_loss, train_acc = train_op(self.idx, self.model, self.train_dataloader, self.optimizer, epochs)


class Server(Node):
    def __init__(self, loader):
        super().__init__(loader)
        self.idx = 0
        self.valid_dataloader = loader

    def merge(self, clients):
        reduce_add_average(target=self.weight, sources=[client.gradient for client in clients])

    def save_model(self, path=None, name=None, verbose=True):
        if name:
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(self.model.state_dict(), os.path.join(path, name))
            if verbose:
                print("Saved model to", os.path.join(path, name))

    def load_model(self, path=None, name=None, verbose=True):
        if name:
            self.model.load_state_dict(torch.load(os.path.join(path, name)))
            if verbose:
                print("Loaded model from", os.path.join(path, name))
