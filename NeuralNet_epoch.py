import os
import numpy as np
import torch
import random
import time
import timeit
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from tabulate import tabulate
from torch import nn, optim
import Config as cfg
from Model_StatsLogger import Model_StatsLogger


class NeuralNet:
    def __init__(self, arch, dataset, epochs, compute_flavour, seed,
                      LR, LRD, WD, MOMENTUM, GAMMA, MILESTONES, device, verbose, gpus, distributed, save_all_states, model_path):

        for gpu_num in range(gpus):
            cfg.LOG.write('NeuralNet __init__: arch={}, dataset={} compute_flavour={}, epochs={},'
                          'LR={} LRD={} WD={} MOMENTUM={} GAMMA={} MILESTONES={} '
                          'device={} verbose={} gpus={} distributed={} model_path={}'
                          .format(arch, dataset, compute_flavour, epochs,
                                  LR, LRD, WD, MOMENTUM, GAMMA, MILESTONES, device, verbose, gpus, distributed, model_path), terminal=(gpu_num == 0), gpu_num=gpu_num)
            cfg.LOG.write('Seed = {}'.format(seed), terminal=(gpu_num == 0), gpu_num=gpu_num)

        if device =='cpu':
            self.device = torch.device('cpu')
        elif device =='cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            cfg.LOG.write('WARNING: Found no valid GPU device - Running on CPU')

        self.compute_flavour = compute_flavour
        self.epochs = epochs
        self.device = device
        self.verbose = verbose
        self.convs_number = 0
        self.LR = LR
        self.LRD = LRD
        self.WD = WD
        self.MOMENTUM = MOMENTUM
        self.GAMMA = GAMMA
        self.MILESTONES = MILESTONES
        self.gpus = gpus
        self.distributed = distributed
        self.model_path = model_path
        self.save_all_states = save_all_states

        torch.manual_seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        #torch.backends.cudnn.enabled = False

        self.arch = '{}_{}'.format(arch, dataset)
        self.dataset = dataset
        self.criterion = nn.CrossEntropyLoss()
        self.criterion = self.criterion.cuda() if device == 'cuda' else self.criterion


        self.model = cfg.MODELS[self.arch](compute_flavour=compute_flavour, device=device, verbose=verbose)
        self.model = self.model.cuda() if device == 'cuda' else self.model
        self.model_stats = Model_StatsLogger(compute_flavour, seed, verbose)
        self.model_optimizer = optim.SGD(self.model.parameters(), lr=LR, weight_decay=WD, momentum=MOMENTUM)
        self.model_train_scheduler = optim.lr_scheduler.MultiStepLR(self.model_optimizer, milestones=MILESTONES, gamma=GAMMA)

        for gpu_num in range(gpus):
            graphs_path = os.path.join(cfg.LOG.graph_path[gpu_num], '{}_CM_Conv'.format(compute_flavour))
            os.mkdir('{}'.format(graphs_path))

        self.load_models()


    def load_models(self, gpu=0, disributed = 0):
        if self.model_path is not None:
            if os.path.isfile(self.model_path):
                chkp = torch.load(self.model_path)
            else:
                assert 0, 'Error: Cannot find model path {}'.format(self.model_path)

            assert (self.arch == chkp['arch'])
            try:
                if disributed == 0:
                    self.model.load_state_dict(chkp['state_dict'], strict=True)
                else:
                    self.model.module.load_state_dict(chkp['state_dict'], strict=True)
                self.model = self.model.cuda() if self.device == 'cuda' else self.model
                self.model_optimizer.load_state_dict(chkp['optimizer'])
                self.model_train_scheduler.load_state_dict(chkp['scheduler'])
                cfg.LOG.write('Loaded model successfully with {} Compute flavour{}'.format(
                    self.model.compute_flavour if disributed == 0 else self.model.module.compute_flavour,
                    '' if disributed == 0 else ' in distributed mode'), terminal=(gpu == 0), gpu_num=gpu)

                cfg.LOG.write('Loaded models successfully{}'.format('' if disributed == 0 else ' in distributed mode'), terminal=(gpu == 0), gpu_num=gpu)
            except RuntimeError as e:
                cfg.LOG.write('Loading model state warning, please review', terminal=(gpu == 0), gpu_num=gpu)
                cfg.LOG.write('{}'.format(e), terminal=(gpu == 0), gpu_num=gpu)

    def print_verbose(self, msg, v):
        if self.verbose >= v:
            cfg.LOG.write(msg)


    def update_batch_size(self, train_set_size, test_set_size):
        self.model_stats.progress['train'].update_batch_num(train_set_size)
        self.model_stats.progress['test'].update_batch_num(test_set_size)

    def reset_accuracy_logger(self, mode):
        self.model_stats.losses[mode].reset()
        self.model_stats.top1[mode].reset()
        self.model_stats.top5[mode].reset()

    def switch_to_train_mode(self):
        # switch to train mode
        self.model.train()

    def switch_to_test_mode(self):
        self.model.eval()


    def log_data_time(self, end, mode):
        # switch to train mode
        self.model_stats.data_time[mode].update(time.time() - end)

    def log_batch_time(self, end, mode):
        # switch to train mode
        self.model_stats.batch_time[mode].update(time.time() - end)


    def compute_forward(self, images, epoch):
        model_out = self.model(images, epoch)
        return model_out

    def compute_loss(self,model_out, target):
        model_loss = self.criterion(model_out, target)
        return model_loss

    def measure_accuracy_log(self, model_out, model_loss, target, images_size, topk, mode):
        acc1, acc5 = self.model_stats.accuracy(model_out, target, topk)
        self.model_stats.losses[mode].update(model_loss.item(), images_size)
        self.model_stats.top1[mode].update(acc1[0], images_size)
        self.model_stats.top5[mode].update(acc5[0], images_size)


    def zero_gradients(self):
        self.model_optimizer.zero_grad()

    def backward_compute(self, model_loss):
        model_loss.backward()


    def compute_step(self):
        self.model_optimizer.step()


    def print_progress(self, epoch, batch, mode, gpu_num):
        self.model_stats.progress[mode].print(' {} Compute flavour conv'.format(self.model_stats.compute_flavour), epoch, batch, gpu_num)


    def log_history(self, epoch, mode='train'):
        self.model_stats.log_history(epoch, mode)


    def set_learning_rate(self):
        if self.LRD == 1:
            self.model_train_scheduler.step()


    def print_epoch_stats(self, epoch, mode='train', gpu_num=0):
        if mode == 'train':
            cfg.LOG.write_title("Training Epoch {} Stats".format(epoch), terminal=(gpu_num == 0), gpu_num=gpu_num)
        elif mode == 'test':
            cfg.LOG.write_title("Testing Epoch {} Stats".format(epoch), terminal=(gpu_num == 0), gpu_num=gpu_num)
        else:
            raise NotImplementedError

        stats_headers = ["Conv", "Avg. Loss", "Avg. Acc1", "Avg. Acc5"]
        stats = []
        stats.append(("{} Compute flavour conv".format(self.model_stats.compute_flavour), self.model_stats.losses[mode].getAverage(),
                      self.model_stats.top1[mode].getAverage(),
                      self.model_stats.top5[mode].getAverage()))
        cfg.LOG.write(tabulate(stats, headers=stats_headers, tablefmt="grid"), date=False, terminal=(gpu_num == 0), gpu_num=gpu_num)


    def _save_state(self, epoch, best_top1_acc, model, optimizer, scheduler, desc):
        if desc is None:
            filename = '{}_epoch-{}_top1-{}.pth'.format(self.arch, epoch, round(best_top1_acc, 2))
        else:
            filename = '{}_epoch-{}_{}_top1-{}.pth'.format(self.arch, epoch, desc, round(best_top1_acc, 2))
        path = '{}/{}'.format(cfg.LOG.models_path, filename)

        state = {'arch': self.arch,
                 'epoch': epoch + 1,
                 'state_dict': model.module.state_dict() if self.distributed == 1 else model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(),
                 'best_top1_acc': best_top1_acc}

        torch.save(state, path)

    def update_best_acc(self, epoch):
        if epoch >= 80 or self.save_all_states == 1:
            top1_acc = self.model_stats.top1['test'].avg
            if top1_acc > self.model_stats.best_top1_acc:
                self.model_stats.best_top1_acc = top1_acc
                self.model_stats.best_top1_epoch = epoch
                self._save_state(epoch=epoch, best_top1_acc=top1_acc.item(), model=self.model, optimizer=self.model_optimizer, scheduler=self.model_train_scheduler,desc='{}_Compute_flavour_Conv'.format(self.model_stats.compute_flavour))


    def export_stats(self, gpu = 0):
        #export stats results
        self.model_stats.export_stats(gpu=gpu)


    def plot_results(self, gpu = 0):
        #plot results for each convolution
        self.model_stats.plot_results(gpu=gpu)


    def distribute_model(self, gpu):
        self.criterion = self.criterion.cuda(gpu)
        self.model.cuda(gpu)
        self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[gpu])
        self.model_optimizer = optim.SGD(self.model.parameters(), lr=self.LR, weight_decay=self.WD, momentum=self.MOMENTUM)
        self.model_train_scheduler = optim.lr_scheduler.MultiStepLR(self.model_optimizer, milestones=self.MILESTONES, gamma=self.GAMMA) # learning rate decay

        self.load_models(gpu=gpu, disributed=1)


    def train(self, epoch, train_gen, gpu = 0):
        cfg.LOG.write_title('Training 
                            {}'.format(epoch), terminal=(gpu==0), gpu_num=gpu)

        if gpu == 0:
            self.print_verbose('NeuralNet train() epoch={}'.format(epoch), 2)
        self.reset_accuracy_logger('train')
        self.switch_to_train_mode()

        end = time.time()
        torch.cuda.synchronize()
        start = timeit.default_timer()

        for i, (images, target) in enumerate(train_gen):
            # measure data loading time

            self.log_data_time(end, 'train')

            if self.device == 'cuda':
                images = images.cuda(non_blocking=True, device=gpu)
                target = target.cuda(non_blocking=True, device=gpu)

            model_out = self.compute_forward(images, epoch)

            model_loss = self.compute_loss(model_out, target)

            # measure accuracy and record logs
            self.measure_accuracy_log(model_out, model_loss, target, images.size(0), topk=(1, 5), mode='train')


            # compute gradient and do SGD step
            self.zero_gradients()

            self.backward_compute(model_loss)

            self.compute_step()

            # measure elapsed time
            self.log_batch_time(end, mode='train')

            end = time.time()

            if i % cfg.BATCH_SIZE == 0:
                self.print_progress(epoch, i, mode='train', gpu_num=gpu)

        self.set_learning_rate()
        torch.cuda.synchronize()
        stop = timeit.default_timer()
        self.log_history(epoch, mode='train')
        self.print_epoch_stats(epoch=epoch, mode='train', gpu_num=gpu)
        cfg.LOG.write('Total Epoch {} Time: {:6.2f} seconds'.format(epoch,stop - start), terminal=(gpu == 0), gpu_num=gpu)

        return


    def test_set(self, epoch, test_gen, gpu = 0):
        cfg.LOG.write_title('Testing Epoch {}'.format(epoch), terminal=(gpu==0), gpu_num=gpu)

        if gpu == 0:
            self.print_verbose('NeuralNet test_set() epoch={}'.format(epoch), 2)
        self.reset_accuracy_logger('test')
        self.switch_to_test_mode()

        with torch.no_grad():
            end = time.time()
            torch.cuda.synchronize()
            start = timeit.default_timer()

            for i, (images, target) in enumerate(test_gen):

                self.log_data_time(end, 'test')

                if self.device == 'cuda':
                    images = images.cuda(non_blocking=True, device=gpu)
                    target = target.cuda(non_blocking=True, device=gpu)

                model_out = self.compute_forward(images, epoch)

                model_loss = self.compute_loss(model_out, target)

                # measure accuracy and record logs
                self.measure_accuracy_log(model_out, model_loss, target, images.size(0), topk=(1, 5), mode='test')

                # measure elapsed time
                self.log_batch_time(end, mode='test')

                end = time.time()

                if i % cfg.BATCH_SIZE == 0:
                    self.print_progress(epoch, i, mode='test', gpu_num=gpu)

            torch.cuda.synchronize()
            stop = timeit.default_timer()
            self.log_history(epoch, mode='test')

            self.print_epoch_stats(epoch=epoch, mode='test', gpu_num=gpu)
            cfg.LOG.write('Total Test Time: {:6.2f} seconds'.format(epoch, stop - start), terminal=(gpu == 0), gpu_num=gpu)


            if gpu == 0:
                self.update_best_acc(epoch)
        return
