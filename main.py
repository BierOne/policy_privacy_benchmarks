# @Time : 2021/3/6 12:28
# @Author : BierOne
# @File : main.py
import os, argparse
import torch
import json
import numpy as np
import torch.nn as nn
import torch.optim as optim

from utilities import config, utils, dataset as data
from train.train import run
from model import base_model


def seed_torch(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print("seed: ", seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='temp')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=28, help='random seed')
    parser.add_argument('--gpu', default='0', help='the chosen gpu id')
    parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='what update to use? rmsprop|adamax|adadelta|adam|sgd')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    seed_torch(args.seed)
    print("epochs", args.epochs)
    print("lr", args.lr)
    print("optimizer", args.optimizer)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    train_loader = data.get_loader('train', args.batch_size)
    val_loader = data.get_loader('test', args.batch_size)

    output_path = 'saved_models/{}/{}'.format(config.dataset, args.output)
    utils.create_dir(output_path)
    torch.backends.cudnn.benchmark = True

    embeddings = np.load(os.path.join(config.dataroot, 'glove6b_init_300d.npy'))
    constructor = 'build_baseline_with_dl'
    model = getattr(base_model, constructor)(embeddings).cuda()
    model = nn.DataParallel(model).cuda()

    if args.optimizer == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), rho=0.95, eps=1e-6, weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=args.weight_decay, momentum=0,
                                    centered=False)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    start_epoch = 0
    f1_val_best = 0
    tracker = utils.Tracker()
    model_path = os.path.join(output_path, 'model.pth')
    for epoch in range(start_epoch, args.epochs):
        run(model, train_loader, optimizer, tracker, train=True, prefix='train', epoch=epoch)
        r = run(model, val_loader, optimizer, tracker, train=False, prefix='test', epoch=epoch)
        if r[4].mean() > f1_val_best:
            f1_val_best = r[4].mean()
            best_results = r
    utils.print_results(best_results[2], best_results[3])

if __name__ == '__main__':
    main()