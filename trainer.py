import os
import random

from tensorboardX import FileWriter

import node
from config import cfg
from util import mkdir_p, Recorder, Summary


class Trainer(object):
    def __init__(self, output_dir, dataloaders):
        self.trainloader, self.validloader, self.testloader = dataloaders
        self.train_num = int(cfg.Total.node_num * cfg.Data.frac)
        self.test_num = cfg.Total.node_num - self.train_num
        self.train_clients = [node.Client(k, self.trainloader[k], self.validloader[k]) for k in range(self.train_num)]
        self.test_list = [node.Client(j + self.train_num, self.trainloader[j], self.validloader[j]) for j in
                          range(self.test_num)]
        self.model_dir = os.path.join(output_dir, 'Model')
        self.log_dir = os.path.join(output_dir, 'Log')
        self.server = node.Server(self.testloader)
        mkdir_p(self.model_dir)
        mkdir_p(self.log_dir)
        self.summary_writer = FileWriter(self.log_dir)
        self.recorder = Recorder()

    def train(self):
        # start training
        for rounds in range(cfg.Total.R):
            print(f'==================The {rounds + 1}-th round==================')
            training_client = random.sample(self.train_clients, 4)
            # print(f' The training clients is {training_client}')
            data_sum = 0
            datalist = []
            for client in training_client:
                data_sum += len(client.train_dataloader.dataset)
                data = len(client.train_dataloader.dataset)
                datalist.append(data)

            for idx, client in enumerate(training_client):
                client.fork(self.server)
                agreement = datalist[idx] / data_sum
                metric = client.compute_gradients(cfg.Total.E, agreement)
                self.recorder.validate(client)
                self.recorder.printer(client)
            self.server.merge(training_client)

            print(f'Start personalization testing')
            for idx, client in enumerate(self.test_list):
                client.fork(self.server)
                print(f'Start adapatation process')
                client.adapt(cfg.Total.E)
                print(f'Start validation')
                self.recorder.validate(client)
                self.recorder.printer(client)

            self.recorder.validate(self.server)
            self.recorder.printer(self.server)

        self.recorder.finish()
        Summary()

        # meta_train_error = 0.0
        # meta_train_accuracy = 0.0
        #
        # summary_meta_train_acc = summary.scalar('Meta_train_Accuracy', meta_train_accuracy / 32)
        # summary_meta_train = summary.scalar('Meta_train_Loss', meta_train_error / 32)
        # self.summary_writer.add_summary(summary_meta_train, rounds)
        # self.summary_writer.add_summary(summary_meta_train_acc, rounds)
        # summary_meta_test = summary.scalar('Test_meta_loss', meta_test_error / 10)
        # summary_meta_test_acc = summary.scalar('Test_meta_accuracy', meta_test_accuracy / 10)
        # self.summary_writer.add_summary(summary_meta_test, rounds)
        # self.summary_writer.add_summary(summary_meta_test_acc, rounds)
