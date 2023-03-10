import argparse
import math
import os
import sys
import random

import matplotlib
import torch.multiprocessing as mp
import torch.distributed as dist

matplotlib.use('Agg')
import torch

import Config as cfg
from NeuralNet import NeuralNet

parser = argparse.ArgumentParser(description='Samer Kurzum, samer152@gmail.com',
                                 formatter_class=argparse.RawTextHelpFormatter)
model_names = ['lenet5-cifar10',
               'alexnet-cifar10',
               'alexnet-cifar100',
               'resnet18-cifar100',
               'resnet18-imagenet'
               ]

parser.add_argument('-a', '--arch', metavar='ARCH', choices=model_names, required=False,
                    help='model architectures and datasets:\n' + ' | '.join(model_names))
parser.add_argument('--action', choices=['TRAINING'], required=True,
                    help='TRAINING: Run given model on a given dataset')
parser.add_argument('--desc')
parser.add_argument('--batch_size', default=128, type=int, metavar='N',
                    help='batch size (default: 128)')
parser.add_argument('--epochs', default=10, type=int,
                    help='number of epochs in training mode')
parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda',
                    help='device to run on')
parser.add_argument('--LR', default=0.1, type=float,
                    help='starting learning rate')
parser.add_argument('--LRD', default=0, type=int,
                    help='learning rate decay - if enabled LR is decreased')
parser.add_argument('--WD', default=0, type=float,
                    help='weight decay')
parser.add_argument('--MOMENTUM', default=0, type=float,
                    help='momentum')
parser.add_argument('--GAMMA', default=0.1, type=float,
                    help='gamma')
parser.add_argument('--MILESTONES', nargs='+', default=[60, 120, 160], type=int,
                    help='milestones')
parser.add_argument('--seed', default=42, type=int,
                    help='seed number')
parser.add_argument('--compute_flavour', default=None, type=int,
                    help='type of matmul/conv')
parser.add_argument('--gpu', nargs='+', default=None,
                    help='GPU to run on (default: 0)')
parser.add_argument('--distributed', default=0, type=int,
                    help='DistributedDataParallel')
parser.add_argument('--save_all_states', default=0, type=int, help='save states for every epoch')
parser.add_argument('--model_path', default=None, help='model path to load')
parser.add_argument('--v', default=0, type=int, help='verbosity level (0,1,2) (default:0)')



def distributed_training(gpu, net, dataset_, epochs, batch_size, logger_path, seed):
    rank = gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=net.gpus,
        rank=rank
    )

    cfg.LOG.path = logger_path
    cfg.LOG.gpus = net.gpus
    cfg.LOG._create_log_dir(path=logger_path, gpus=net.gpus, create_logs=False)
    cfg.LOG._update_log_file()

    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.enabled = False
    torch.cuda.set_device(gpu)
    net.distribute_model(gpu)
    test_gen, _ = dataset_.testset(batch_size=batch_size, total_gpus=net.gpus, distributed = True)
    (train_gen, _), (_, _) = dataset_.trainset(batch_size=batch_size, max_samples=None, random_seed=16, gpu=gpu, total_gpus=net.gpus, distributed = True)

    net.update_batch_size(len(train_gen), len(test_gen))

    for epoch in range(0, epochs):
        net.train(epoch, train_gen, gpu)
        net.test_set(epoch, test_gen, gpu)

    net.export_stats(gpu)
    net.plot_results(gpu)



def train_network(arch, dataset, epochs, batch_size, compute_flavour, seed,
                  LR, LRD, WD, MOMENTUM, GAMMA, MILESTONES, device, verbose, distributed, gpus, desc, save_all_states, model_path):

    if seed is None:
        seed = torch.random.initial_seed() & ((1 << 63) - 1)
    name_str = '{}_{}_training_network'.format(arch, dataset)
    name_str = name_str + '_{}'.format(desc) if desc is not None else name_str
    if compute_flavour is not None:
        #assert (1 not in threads or single_thread == False), "Computing 1 thread convolution twice. Please remove 1 from threads list"
        name_str = name_str + '_flavour-{}_epochs-{}'.format(compute_flavour, epochs)
    else:
        name_str = name_str + '_flavour-[{}]_epochs-{}'.format(0, epochs)

    assert (len(gpus) == 1 and distributed == 0) or (len(gpus) > 1 and distributed == 1), 'Error in GPUs numbers in {}Distributed Mode'.format('Non-' if distributed == 0 else '')
    gpus_num = len(gpus) if distributed == 1 else 1
    cfg.LOG.start_new_log(name=name_str, gpus=gpus_num)

    for gpu in range(gpus_num):
        cfg.LOG.write('arch={}, dataset={}, desc={}, flavour={}, epochs={}, batch_size={}, LR={}, LRD={}, WD={}, MOMENTUM={}, GAMMA={}, '
                      'MILESTONES={}, device={}, verbose={}, model_path={}'
                      .format(arch, dataset, desc, compute_flavour, epochs, batch_size, LR, LRD, WD, MOMENTUM, GAMMA, MILESTONES, device, verbose, model_path),
                      terminal=(gpu == 0), gpu_num=gpu)
    cfg.LOG.write('Seed = {}'.format(seed), terminal=(gpu == 0), gpu_num=gpu)
    cfg.LOG.write_title('TRAINING NETWORK', terminal=(gpu == 0), gpu_num=gpu)

    dataset_ = cfg.get_dataset(dataset)

    #build model
    net = NeuralNet(arch, dataset, epochs, compute_flavour, seed,
                    LR, LRD, WD, MOMENTUM, GAMMA, MILESTONES, device, verbose, gpus_num, distributed, save_all_states, model_path)

    if distributed == 0:

        #NORMAL TRAINING
        test_gen, _ = dataset_.testset(batch_size=batch_size)
        (train_gen, _), (_, _) = dataset_.trainset(batch_size=batch_size, max_samples=None, random_seed=16)
        net.update_batch_size(len(train_gen), len(test_gen))
        for epoch in range(0, epochs):
                net.train(epoch, train_gen)
                net.test_set(epoch, test_gen)

        net.export_stats()
        net.plot_results()
    else:

        #distributed training
        cfg.LOG.close_log()
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        mp.spawn(distributed_training, nprocs=len(gpus), args=(net, dataset_, epochs, batch_size, cfg.LOG.path, seed))


def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)

    cfg.BATCH_SIZE = args.batch_size
    assert math.floor(math.log2(args.batch_size)) == math.ceil(math.log2(args.batch_size)) , 'Error: Implementation supports only batch size which is 2^x'

    cfg.USER_CMD = ' '.join(sys.argv)

    if args.action == 'TRAINING':
        assert (args.arch is not None), "Please provide an ARCH name to execute training on"
        arch = args.arch.split('-')[0]
        dataset = args.arch.split('-')[1]

        train_network(arch, dataset, epochs=args.epochs, batch_size=args.batch_size, compute_flavour=args.compute_flavour, seed=args.seed,
                              LR=args.LR, LRD=args.LRD, WD=args.WD, MOMENTUM=args.MOMENTUM,
                              GAMMA=args.GAMMA, MILESTONES=args.MILESTONES,
                              device=args.device, verbose=args.v, distributed=args.distributed, gpus=[int(x) for x in args.gpu],
                              desc=args.desc, save_all_states=args.save_all_states, model_path=args.model_path)
    else:
        raise NotImplementedError



if __name__ == '__main__':
    main()
