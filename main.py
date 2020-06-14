import argparse
import datetime
import pprint
import random
import time

import dateutil.tz

from config import cfg, cfg_from_file
from data import Data
from trainer import Trainer
from util import setup_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('--cfg', dest='cfg_file', default='cfg/fed_avg.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='-1')
    parser.add_argument('--data_dir', type=str, default='~/data')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # config initialization
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.gpu_id != '-1':
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir

    print('Using config')
    pprint.pprint(cfg)

    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    setup_seed(args.manualSeed)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = f'./output/{cfg.Data.dataset}_{cfg.config_name}_{timestamp}'

    num_gpu = len(cfg.GPU_ID.split(','))
    data = Data()
    dataset = (data.trainloader, data.validloader, data.testloader)

    algo = Trainer(output_dir, dataset)
    start_t = time.time()
    algo.train()

    end_t = time.time()
    print(f'Total time for training:{end_t - start_t}')
