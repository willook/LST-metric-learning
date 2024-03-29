#!/usr/bin/env python
from __future__ import print_function

import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
from model.baseline import TextCLIP
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import glob
from functools import partial
import datetime
from pathlib import Path
import matplotlib.pyplot as plt

# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

import utils
from torchlight import DictAction
from tools import *
from Text_Prompt import *
from losses import KLLoss, KLLoss2, Proxy_Anchor
from pytorch_metric_learning.losses import SupConLoss, MultiSimilarityLoss

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

classes, num_text_aug, text_dict = text_prompt_openai_pasta_pool_4part() # text_dict: id(0~4), [120, 77]
text_list = text_prompt_openai_random() # 120, 11, 1, 77

device = "cuda" if torch.cuda.is_available() else "cpu"
scaler = torch.cuda.amp.GradScaler()
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    parser.add_argument(
        '--remark',
        default=None,
        help='Prefix for logdir name')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=10,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--save-epoch',
        type=int,
        default=0,
        help='the start epoch to save model (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=0,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for test')
    parser.add_argument(
        '--examplar-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for examplar')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        action=DictAction,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=32, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=32, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.01,
        help='weight decay for optimizer')
    parser.add_argument(
        '--lr-decay-rate',
        type=float,
        default=0.1,
        help='decay rate for learning rate')
    parser.add_argument('--warm_up_epoch', type=int, default=0)
    parser.add_argument('--loss-alpha', type=float, default=0.8)
    parser.add_argument('--te-lr-ratio', type=float, default=1)
    parser.add_argument('--te-trainable', action='store_true', default=False)
    parser.add_argument('--method', type=str, default='mixing')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--loss', type=str, default="supcon")
    parser.add_argument('--num_text_aug', type=int, default=None)
    return parser


class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.edit_arg()
        self.save_arg()
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    # answer = input('delete it? y/n:')
                    # if answer == 'y':
                    shutil.rmtree(arg.model_saved_name)
                    print('Dir removed: ', arg.model_saved_name)
                    # input('Refresh the website of tensorboard by pressing any keys')
                    # else:
                    #     print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0
        # pdb.set_trace()
        print("load model...")
        self.load_model()

        if self.arg.phase == 'model_size':
            pass
        else:
            self.load_optimizer()
            print("load data...")
            self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0

        self.model = self.model.to(device)
        # self.model = self.model.cuda(self.output_device)

        # if type(self.arg.device) is list:
        #     if len(self.arg.device) > 1:
        #         self.model = nn.DataParallel(
        #             self.model,
        #             device_ids=self.arg.device,
        #             output_device=self.output_device)

        # if type(self.arg.device) is list:
        #     if len(self.arg.device) > 1:
        #         for name in self.arg.model_args['head']:
        #             self.model_text_dict[name] = nn.DataParallel(
        #                 self.model_text_dict[name],
        #                 device_ids=self.arg.device,
        #                 output_device=self.output_device)

    def edit_arg(self):
        global num_text_aug
        date = datetime.date.today().isoformat()
        config = self.arg.config.split(".")[0].split("/")[-1]
        model = "_".join(self.arg.model.split("."))
        method = self.arg.method
        loss = self.arg.loss
        remark = self.arg.remark 
        
        if self.arg.num_text_aug is not None:
            num_text_aug = int(self.arg.num_text_aug)
            
        work_dir = f"./work_dir/{date}/{config}_{model}_{method}_{loss}_{num_text_aug}"
        if remark is not None:
            work_dir = f"{work_dir}_{remark}"
        if self.arg.test:
            work_dir = f"./work_dir/{date}/test"
        self.arg.work_dir = Path(work_dir)

        if self.arg.phase == "test":
            self.arg.work_dir = Path(os.path.dirname(self.arg.weights))
        
        # if self.arg.method == "mixing":
        #     self.arg.te_trainable = True
        # elif self.arg.method == "multimodal":
        #     self.arg.te_trainable = True
        # elif self.arg.method == "naive":
        #     self.arg.te_trainable = True
        # else:
        #     raise NotImplementedError
            
    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.test:
            self.arg.train_feeder_args['data_path'] = 'data/ntu_test_os/NTU_ALL_OS_002.npz'
            self.arg.test_feeder_args['data_path'] = 'data/ntu_test_os/NTU_ALL_OS_002.npz'
            self.arg.examplar_feeder_args['data_path'] = 'data/ntu_test_os/NTU_ALL_OS_002.npz'
            # self.arg.train_feeder_args['data_path'] = 'data_raw/ntu_all_os/NTU_ALL_OS.npz'
            # self.arg.test_feeder_args['data_path'] = 'data_raw/ntu_all_os/NTU_ALL_OS.npz'
            # self.arg.examplar_feeder_args['data_path'] = 'data_raw/ntu_all_os/NTU_ALL_OS.npz'
            
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        else:
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)
        self.data_loader['examplar'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.examplar_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):        
        # output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)
        self.model = Model(**self.arg.model_args)
        # print(self.model)
        self.loss_ce = nn.CrossEntropyLoss().to(device)
        self.loss_ce_img = nn.CrossEntropyLoss().to(device)
        self.loss_ce_text = nn.CrossEntropyLoss().to(device)
        self.loss_img = KLLoss().to(device) # multimodal image
        self.loss_text = KLLoss().to(device) # multimodal text
        self.loss_kl = KLLoss2().to(device)
        # TODO 아래로 바꾸기
        # self.loss_kl = KLLoss3().to(device) # for mixing teacher - student

        self.loss_recons = torch.nn.MSELoss()
        
        self.proxy = False
        loss = self.arg.loss.lower()
        if loss == "proxy_anchor":
            self.loss_me_s = Proxy_Anchor(nb_classes = self.arg.model_args['num_class'], 
                                        sz_embed = 512, mrg = 0.1, alpha = 32).cuda()
            self.loss_me_t = Proxy_Anchor(nb_classes = self.arg.model_args['num_class'], 
                                        sz_embed = 512, mrg = 0.1, alpha = 32).cuda()
            self.proxy = True
        elif loss == "supcon":
            self.loss_me_s = SupConLoss(temperature=0.1)
            self.loss_me_t = SupConLoss(temperature=0.1)
        elif loss == "ms":
            self.loss_me_s = MultiSimilarityLoss(alpha=2, beta=50, base=0.5)
            self.loss_me_t = MultiSimilarityLoss(alpha=2, beta=50, base=0.5)
        else:
            raise NotImplementedError
        self.model_text_dict = nn.ModuleDict()

        for name in self.arg.model_args['head']:
            # name: ViT-B/32
            model_, preprocess = clip.load(name, device)
            # model_, preprocess = clip.load('ViT-L/14', device)
            del model_.visual
            if self.arg.method == "multimodal":
                model_text = TextCLIP(model_, additional_layers=True, name=name)
            elif self.arg.method == "mixing":
                model_text = TextCLIP(model_, additional_layers=True, name=name, mixing=True)
            else:
                model_text = TextCLIP(model_, additional_layers=True, name=name)
            model_text = model_text.to(device)
            self.model_text_dict[name] = model_text
        
        if self.arg.weights:
            self.global_step = int(arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict([[k.split('module.')[-1], v.to(device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

            

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            param_group = [{'params': self.model.parameters(),'lr': self.arg.base_lr}]
            if self.proxy:
                param_group.append({'params': self.loss_me_s.parameters(), 'lr': self.arg.base_lr})
                param_group.append({'params': self.loss_me_t.parameters(), 'lr': self.arg.base_lr})
                
            if self.arg.te_trainable:
                param_group.append({'params': self.model_text_dict.parameters(), 'lr': self.arg.base_lr*self.arg.te_lr_ratio})
            else:
                name = self.arg.model_args['head'].__getitem__(0)
                param_group.append({'params': self.model_text_dict[name].linear_layer.parameters(), 'lr': self.arg.base_lr})
            
            self.optimizer = optim.SGD(
                param_group,
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                [{'params': self.model.parameters(),'lr': self.arg.base_lr},
                {'params': self.loss_me.parameters(), 'lr': self.arg.base_lr}],
                #self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        
        self.adjust_learning_rate(epoch)

        loss_value = []
        acc_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, ncols=40)


        for batch_idx, (data, label, index) in enumerate(process):   
            self.global_step += 1
            with torch.no_grad():
                # data = data.float().cuda(self.output_device)
                data = data.float().to(device)
            timer['dataloader'] += self.split_time()
            self.optimizer.zero_grad()

            # forward
            with torch.cuda.amp.autocast():
                output, feature_dict, logit_scale, part_feature_list = self.model(data)
                label_g = gen_label(label) # n by n 같은 라벨 표기
                label = label.long().to(device)
                loss_te_list = []
                for ind in range(num_text_aug):
                    if ind > 0: # 0번 외에는 바디 파트 스크립트를 넣어줌
                        text_id = np.ones(len(label),dtype=np.int8) * ind # 몇번째 바디 파트인지
                        texts = torch.stack([text_dict[j][i,:] for i,j in zip(label,text_id)])
                        texts = texts.to(device)

                    else:
                        # 0번은 11개의 동의어 중 랜덤 라벨을 넣어줌
                        texts = list()
                        for i in range(len(label)):
                            text_len = len(text_list[label[i]])
                            text_id = np.random.randint(text_len,size=1)
                            text_item = text_list[label[i]][text_id.item()]
                            texts.append(text_item)

                        texts = torch.cat(texts).to(device)
                    
                    if self.arg.method == "mixing":
                        text_embedding, mixed_embedding = self.model_text_dict[self.arg.model_args['head'][0]](texts, 
                                            feature_dict[self.arg.model_args['head'][0]])
                        text_embedding = text_embedding.float()
                        mixed_embedding = mixed_embedding.float()
                    else:
                        text_embedding = self.model_text_dict[self.arg.model_args['head'][0]](texts).float()

                    if ind == 0:
                        # logit: (100 x 512)
                        logits_per_image, logits_per_text = create_logits(feature_dict[self.arg.model_args['head'][0]],text_embedding,logit_scale[:,0].mean())

                        ground_truth = torch.tensor(label_g,dtype=feature_dict[self.arg.model_args['head'][0]].dtype,device=device)
                    else:
                        logits_per_image, logits_per_text = create_logits(part_feature_list[ind-1],text_embedding,logit_scale[:,ind].mean())

                        ground_truth = torch.tensor(label_g,dtype=part_feature_list[ind-1].dtype,device=device)

                    if self.arg.method == "mixing":
                        loss_metric = self.loss_me_s(feature_dict[self.arg.model_args['head'][0]], label)
                        loss_metric_mix = self.loss_me_t(mixed_embedding, label)
                        loss_ts = self.loss_kl(feature_dict[self.arg.model_args['head'][0]], mixed_embedding)
                        loss_te_list.append(loss_ts)
                        loss_te_list.append(loss_metric)
                        loss_te_list.append(loss_metric_mix)
                    elif self.arg.method == "multimodal":
                        loss_metric_img = self.loss_me_s(feature_dict[self.arg.model_args['head'][0]], label)
                        #loss_metric_txt = self.loss_me_t(text_embedding, label)
                        loss_imgs = self.loss_img(logits_per_image,ground_truth) # (batch size x batch size) KLLoss -> why?
                        loss_texts = self.loss_text(logits_per_text,ground_truth)
                        loss_te_list.append((loss_imgs + loss_texts) / 2)
                        loss_te_list.append(loss_metric_img)
                        #loss_te_list.append(loss_metric_txt)
                    elif self.arg.method == "naive":
                        loss_metric = self.loss_me_s(feature_dict[self.arg.model_args['head'][0]], label)
                        loss_te_list.append(loss_metric)
                    else:
                        raise NotImplementedError
                
                loss_ce = self.loss_ce(output, label)
                loss = (loss_ce + self.arg.loss_alpha * sum(loss_te_list) / len(loss_te_list))
                
                # reconstructed = feature_dict['reconstructed'] * data.bool()
                # loss_recons = self.loss_recons(data, reconstructed) 
                # loss += loss_recons
                
            self.losses_list.append(loss.data.cpu())
            scaler.scale(loss).backward()

            scaler.step(self.optimizer)
            scaler.update()

            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()

            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            acc_value.append(acc.data.item())
            self.train_writer.add_scalar('acc', acc, self.global_step)
            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)
            
            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

            # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.  Mean training acc: {:.2f}%.'.format(np.mean(loss_value), np.mean(acc_value)*100))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None, save_model=False):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        
        for ln in loader_name:
            recalls, accuracy, test_embeddings, test_labels, acc_per_class = utils.evaluate_one_shot(
                self.model, self.data_loader['test'], self.data_loader['examplar'])
            self.acc_per_class = acc_per_class
            self.accuracy_list.append(accuracy)
            if accuracy > self.best_acc or save_model:
                self.save_model(epoch)
            format = "png"
            
            if self.arg.phase == 'test':
                train_embeddings, train_labels = utils.predict_batchwise(self.model, self.data_loader['train'])
                utils.save_tsne_plot(train_embeddings, train_labels, self.arg.work_dir, 
                                     format=format, phase="train")
                utils.save_tsne_subplots(train_embeddings, train_labels, 
                                    test_embeddings, test_labels, self.arg.work_dir, format=format)
                #train_embeddings = l2_norm(train_embeddings)

                _saved_path = arg.weights.split('.')[0]
                saved_path = f'saved_embeddings/{_saved_path}'
                os.makedirs(saved_path, exist_ok=True)
                torch.save({'train': [train_embeddings, train_labels],
                            'test': [test_embeddings, test_labels]},
                            f'{saved_path}/embeddings.pt')
                print(f"Save the embeddings as {saved_path}/embeddings.pt")
                
            if accuracy > self.best_acc:
                utils.save_tsne_plot(test_embeddings, test_labels, self.arg.work_dir, 
                                     format=format, phase=epoch)
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1
                acc_per_class_str = " ".join([f"{acc:.4f}" for acc in acc_per_class])
                self.print_log(f'\tStd per Class: {np.std(acc_per_class)}')
                self.print_log(f'\tACC per Class: {acc_per_class_str}')
                
                #print(f'\tAcc per Class: {acc_per_class_str}')
            
            #print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
            #print('best_acc: ', self.best_acc, ' epoch: ', self.best_acc_epoch)
            self.print_log(f'\tAccuracy: {accuracy},  model: {self.arg.model_saved_name}')
            self.print_log(f'\tBest Acc: {self.best_acc} epoch: {self.best_acc_epoch}')

            if self.arg.phase == 'train':
                # self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)

    def save_plot(self):
        plt.plot(self.accuracy_list)
        plt.savefig(f"{self.arg.work_dir}/accurcy_plot.png")
        plt.clf()
        plt.plot(self.losses_list)
        plt.savefig(f"{self.arg.work_dir}/loss_plot.png")
        plt.clf()
        
    def save_model(self, epoch):
        state_dict = self.model.state_dict()
        weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
        torch.save(weights, self.arg.model_saved_name + '-' + str(epoch+1) + '-' + str(int(self.global_step)) + '.pt')
    
    def start(self):
        print("start the processer")
        self.accuracy_list = []
        self.losses_list = []
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print_log(f'# Parameters: {count_parameters(self.model)}')
            start_epoch = 0
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = (((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)) and (epoch+1) > self.arg.save_epoch
                self.train(epoch)
                self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'], save_model=save_model)
                self.save_plot()
            # test the best model
            weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-'+str(self.best_acc_epoch)+'*'))[0]
            #weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs'))[0]
            weights = torch.load(weights_path)
            if type(self.arg.device) is list:
                if len(self.arg.device) > 1:
                    weights = OrderedDict([['module.'+k, v.to(device)] for k, v in weights.items()])
            self.model.load_state_dict(weights)

            wf = weights_path.replace('.pt', '_wrong.txt')
            rf = weights_path.replace('.pt', '_right.txt')
            self.arg.print_log = False
            self.eval(epoch=0, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.arg.print_log = True


            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')

        elif self.arg.phase == 'test':
            wf = self.arg.weights.replace('.pt', '_wrong.txt')
            rf = self.arg.weights.replace('.pt', '_right.txt')

            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            #utils.save_tsne_plot(embeddings, labels, root, num_classes=121, epoch=0)
            self.print_log('Done.\n')

if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    print(p.config)
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:   
                print('WRONG ARG: {}'.format(k))
                assert (k in key), f"{k} not in key"
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    #device = [str(device_id) for device_id in arg.device]
    #os.environ["CUDA_VISIBLE_DEVICES"]= ", ".join(device)
    #os.environ["CUDA_VISIBLE_DEVICES"]= str(arg.device)
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()
