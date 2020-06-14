import random

import numpy as np
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms

from config import cfg


def get_dataset():
    train_dataset, test_dataset = None, None

    if cfg.Data.dataset == 'cifar10':
        data_dir = '~/data/cifar/'
        tra_trans = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        val_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=tra_trans)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                        transform=val_trans)

    return train_dataset, test_dataset


def print_split(idcs, labels):
    n_labels = np.max(labels) + 1
    print("Data split:")
    for i, idccs in enumerate(idcs):
        if i < 10 or i > len(idcs) - 10:
            split = np.sum(np.array(labels)[idccs].reshape(1, -1) == np.arange(n_labels).reshape(-1, 1), axis=1)
            print(" - Client {}: {}".format(i, split), flush=True)
        elif i == len(idcs) - 10:
            print(".  " * 10 + "\n" + ".  " * 10 + "\n" + ".  " * 10)
    print()


class Data(object):

    def __init__(self):
        self.frac = cfg.Data.frac
        self.batchsize = cfg.Data.batchsize
        self.dataset = cfg.Data.dataset
        self.num_users = cfg.Total.node_num
        self.sampling_mode = cfg.Data.sampling_mode
        self.equal = cfg.Data.equal
        self.trainset, self.testset = get_dataset()
        self.train_dict_users = self.sample(self.trainset, self.num_users, self.sampling_mode, self.equal)
        self.max_users = max(int(self.frac * self.num_users), 1)
        self.train_users = np.random.choice(range(self.num_users), self.max_users, replace=False)
        self.test_users = np.array(list(set(range(self.num_users)) - set(self.train_users)))
        self.trainloader, self.validloader, self.testloader = self.splitdatset(self.num_users)

        # num_train = [int(len(trainset) / args.split) for _ in range(args.split)]
        # cumsum_train = torch.tensor(list(num_train)).cumsum(dim=0).tolist()
        # # idx_train = sorted(range(len(trainset.targets)), key=lambda k: trainset.targets[k])  #split by class
        # idx_train = range(len(trainset.targets))
        # splited_trainset = [Subset(trainset, idx_train[off - l:off]) for off, l in zip(cumsum_train, num_train)]
        # num_test = [int(len(testset) / args.split) for _ in range(args.split)]
        # cumsum_test = torch.tensor(list(num_test)).cumsum(dim=0).tolist()
        # # idx_test = sorted(range(len(testset.targets)), key=lambda k: trainset.targets[k])  #split by class
        # idx_test = range(len(testset.targets))
        # splited_testset = [Subset(testset, idx_test[off - l:off]) for off, l in zip(cumsum_test, num_test)]
        # self.test_all = DataLoader(testset, batch_size=args.batchsize, shuffle=False, num_workers=4)
        # self.train_loader = [DataLoader(splited_trainset[i], batch_size=args.batchsize, shuffle=True, num_workers=4)
        #                      for i in range(args.node_num)]
        # self.test_loader = [DataLoader(splited_testset[i], batch_size=args.batchsize, shuffle=False, num_workers=4)
        #                     for i in range(args.node_num)]
        # self.test_loader = DataLoader(testset, batch_size=args.batchsize, shuffle=False, num_workers=4)

    def sample(self, dataset, num_users, sampling_mode, equal):
        if sampling_mode == 'iid':
            num_items = int(len(dataset) / num_users)
            dict_users, all_idxs = {}, [i for i in range(len(dataset))]
            for i in range(num_users):
                dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
                all_idxs = list(set(all_idxs) - dict_users[i])
            return dict_users
        else:
            if equal:
                all_idxs = np.random.permutation(len(dataset))
                dict_users = {}
                labels = np.array(dataset.targets)
                alpha = cfg.Data.alpha
                n_classes = np.max(labels) + 1
                # return num_users * classes
                # label_distribution = np.random.dirichlet([alpha] * n_classes, num_users).transpose()
                label_distribution = np.random.dirichlet([alpha] * num_users, n_classes)
                class_idxes = [np.argwhere(labels[all_idxs] == y) for y in range(n_classes)]
                client_idxs = [[] for _ in range(num_users)]
                for c, fracs in zip(class_idxes, label_distribution):
                    for i, idx in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
                        client_idxs[i] += [idx]

                client_idxs = [all_idxs[np.concatenate(idx)] for idx in client_idxs]
                for i in range(num_users):
                    dict_users[i] = client_idxs[i]
                print_split(client_idxs, labels)

                return dict_users

    # def sampling(self, dataset, num_users, sampling_mode, equal):
    #     if sampling_mode == 'iid':
    #         num_items = int(len(dataset) / num_users)
    #         dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    #         for i in range(num_users):
    #             dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
    #             all_idxs = list(set(all_idxs) - dict_users[i])
    #         return dict_users
    #     else:
    #         if equal:
    #             all_samples = len(dataset)
    #             num_imgs = 200
    #             num_shards = int(all_samples / num_imgs)
    #             idx_shard = [i for i in range(num_shards)]
    #             dict_users = {i: np.array([]) for i in range(num_users)}
    #             idxs = np.arange(num_shards * num_imgs)
    #             labels = np.array(dataset.targets)
    #
    #             # sort labels
    #             idxs_labels = np.vstack((idxs, labels))
    #             idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    #             idxs = idxs_labels[0, :]
    #
    #             # dived and assign x shards/client
    #             x = int(num_shards / num_users)
    #             for i in range(num_users):
    #                 # length = len(idx_shard)
    #                 rand_set = set(np.random.choice(idx_shard, x, replace=False))
    #                 # flag = int(x / 3)
    #                 # rand_set_1 = set(idx_shard[:flag])
    #                 # rand_set_2 = set(idx_shard[int(length/3):int(length/3)+flag])
    #                 # rand_set_3 = set(idx_shard[-flag:])
    #                 # rand_set = rand_set_1 | rand_set_2|rand_set_3
    #                 idx_shard = list(set(idx_shard) - rand_set)
    #                 for rand in rand_set:
    #                     dict_users[i] = np.concatenate(
    #                         (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0
    #                     )
    #             return dict_users

    def splitdatset(self, idxs_users):
        trainloader, validloader, testloader = [], [], []
        for idx in range(idxs_users):
            train_idxs = list(self.train_dict_users[idx])
            num_train = int(len(train_idxs) * 0.8)
            num_val = len(train_idxs) - num_train
            idx_train = random.sample(train_idxs, num_train)
            idx_valid = random.sample(train_idxs, num_val)
            # idx_train = train_idxs[:int(0.8 * len(train_idxs))]
            # idx_valid = train_idxs[int(0.8 * len(train_idxs)):]
            trainloader.append(
                DataLoader(CustomSubset(self.trainset, idx_train), batch_size=self.batchsize, shuffle=True)
            )
            validloader.append(
                DataLoader(CustomSubset(self.trainset, idx_valid), batch_size=self.batchsize, shuffle=False)
            )
        testloader = DataLoader(self.testset, batch_size=self.batchsize, shuffle=False)

        return trainloader, validloader, testloader


class CustomSubset(Subset):
    def __init__(self, dataset, indices, subset_transform=None):
        super().__init__(dataset, indices)
        self.subset_transform = subset_transform

    def __getitem__(self, idx):
        x, y = self.dataset[np.asscalar(self.indices[idx])]
        if self.subset_transform:
            x = self.subset_transform(x)

        return x, y
