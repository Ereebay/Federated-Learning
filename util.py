import errno
import os
import os.path as osp
import random

import numpy as np
import torch
from torchvision import transforms

from config import cfg


def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def save_image(tensor):
    unloader = transforms.ToPILImage()
    dir = 'results'
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    if not osp.exists(dir):
        os.makedirs(dir)
    image.save('test.jpg')


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


# TODO 各类metric重写
class Recorder(object):
    def __init__(self):
        self.counter = 0
        self.tra_loss = {}
        self.tra_acc = {}
        self.val_loss = {}
        self.val_acc = {}
        for i in range(cfg.Total.node_num + 1):
            self.val_loss[str(i)] = []
            self.val_acc[str(i)] = []
            self.val_loss[str(i)] = []
            self.val_acc[str(i)] = []
        self.acc_best = torch.zeros(cfg.Total.node_num + 1)
        self.get_a_better = torch.zeros(cfg.Total.node_num + 1)

    def validate(self, node):
        self.counter += 1
        node.model.to(node.device).eval()
        total_loss = 0.0
        correct = 0.0

        with torch.no_grad():
            for idx, (data, target) in enumerate(node.valid_dataloader):
                data, target = data.to(node.device), target.to(node.device)
                output = node.model(data)
                total_loss += torch.nn.CrossEntropyLoss()(output, target)
                pred = output.argmax(dim=1)
                correct += pred.eq(target.view_as(pred)).sum().item()
            total_loss = total_loss / (idx + 1)
            acc = correct * 100 / len(node.valid_dataloader.dataset)
        self.val_loss[str(node.idx)].append(total_loss)
        self.val_acc[str(node.idx)].append(acc)

        if self.val_acc[str(node.idx)][-1] > self.acc_best[node.idx]:
            self.get_a_better[node.idx] = 1
            self.acc_best[node.idx] = self.val_acc[str(node.idx)][-1]
            torch.save(node.model.state_dict(),
                       './saves/model/Node{:d}_{:s}.pt'.format(node.idx, cfg.Model.local_model))

    def printer(self, node):
        print(f'Node{node.idx}: Accuracy: {self.val_acc[str(node.idx)][-1]:.2f}%!')
        if self.get_a_better[node.idx] == 1:
            print('Node{:d}: A Better Accuracy: {:.2f}%! Model Saved!'.format(node.idx, self.acc_best[node.idx]))
            print('-------------------------')
            self.get_a_better[node.idx] = 0

    def finish(self):
        torch.save([self.val_loss, self.val_acc],
                   './saves/record/loss_acc_{:s}_{:s}.pt'.format(cfg.Total.alogrithm, cfg.Total.notes))
        print('Finished!\n')
        for i in range(cfg.Total.node_num + 1):
            print('Node{}: Best Accuracy = {:.2f}%'.format(i, self.acc_best[i]))


def LR_scheduler(rounds, Node_List):
    trigger = int(cfg.Total.R / 3)
    if rounds != 0 and rounds % trigger == 0 and rounds < cfg.Optima.stop_decay:
        cfg.Optima.lr *= 0.1
        # args.alpha += 0.2
        # args.beta += 0.4
        for i in range(len(Node_List)):
            Node_List[i].args.lr = cfg.Optima.lr
            Node_List[i].args.alpha = cfg.Optima.alpha
            Node_List[i].args.beta = cfg.Optima.beta
            Node_List[i].optimizer.param_groups[0]['lr'] = cfg.Optima.lr
            Node_List[i].meme_optimizer.param_groups[0]['lr'] = cfg.Optima.lr
    print('Learning rate={:.4f}'.format(cfg.Optima.lr))


def Summary():
    print("Summary：\n")
    print("algorithm:{}\n".format(cfg.Total.alogrithm))
    print("dataset:{}\tbatchsize:{}\n".format(cfg.Data.dataset, cfg.Data.batchsize))
    print("node_num:{},\tsplit:{}\n".format(cfg.Total.node_num, cfg.Data.split))
    # print("iid:{},\tequal:{},\n".format(args.iid == 1, args.unequal == 0))
    print("global epochs:{},\tlocal epochs:{},\n".format(cfg.Total.R, cfg.Total.E))
    print("global_model:{}，\tlocal model:{},\n".format(cfg.Model.global_model, cfg.Model.local_model))
