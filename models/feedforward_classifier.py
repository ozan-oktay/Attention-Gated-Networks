import os
import numpy as np
import utils.util as util
from collections import OrderedDict

import torch
from torch.autograd import Variable
from .base_model import BaseModel
from .networks import get_network
from .layers.loss import *
from .networks_other import get_scheduler, print_network, benchmark_fp_bp_time
from .utils import classification_stats, get_optimizer, get_criterion
from .networks.utils import HookBasedFeatureExtractor


class FeedForwardClassifier(BaseModel):

    def name(self):
        return 'FeedForwardClassifier'

    def initialize(self, opts, **kwargs):
        BaseModel.initialize(self, opts, **kwargs)
        self.opts = opts
        self.isTrain = opts.isTrain

        # define network input and output pars
        self.input = None
        self.target = None
        self.labels = None
        self.tensor_dim = opts.tensor_dim

        # load/define networks
        self.net = get_network(opts.model_type, n_classes=opts.output_nc,
                               in_channels=opts.input_nc, nonlocal_mode=opts.nonlocal_mode,
                               tensor_dim=opts.tensor_dim, feature_scale=opts.feature_scale,
                               attention_dsample=opts.attention_dsample,
                               aggregation_mode=opts.aggregation_mode)
        if self.use_cuda: self.net = self.net.cuda()

        # load the model if a path is specified or it is in inference mode
        if not self.isTrain or opts.continue_train:
            self.path_pre_trained_model = opts.path_pre_trained_model
            if self.path_pre_trained_model:
                self.load_network_from_path(self.net, self.path_pre_trained_model, strict=False)
                self.which_epoch = int(0)
            else:
                self.which_epoch = opts.which_epoch
                self.load_network(self.net, 'S', self.which_epoch)

        # training objective
        if self.isTrain:
            self.criterion = get_criterion(opts)
            # initialize optimizers
            self.schedulers = []
            self.optimizers = []

            self.optimizer = get_optimizer(opts, self.net.parameters())
            self.optimizers.append(self.optimizer)

        # print the network details
        if kwargs.get('verbose', True):
            print('Network is initialized')
            print_network(self.net)

        # for accumulator
        self.reset_results()

    def set_scheduler(self, train_opt):
        for optimizer in self.optimizers:
            self.schedulers.append(get_scheduler(optimizer, train_opt))
            print('Scheduler is added for optimiser {0}'.format(optimizer))

    def set_input(self, *inputs):
        # self.input.resize_(inputs[0].size()).copy_(inputs[0])
        for idx, _input in enumerate(inputs):
            # If it's a 5D array and 2D model then (B x C x H x W x Z) -> (BZ x C x H x W)
            bs = _input.size()
            if (self.tensor_dim == '2D') and (len(bs) > 4):
                _input = _input.permute(0,4,1,2,3).contiguous().view(bs[0]*bs[4], bs[1], bs[2], bs[3])

            # Define that it's a cuda array
            if idx == 0:
                self.input = _input.cuda() if self.use_cuda else _input
            elif idx == 1:
                self.target = Variable(_input.cuda()) if self.use_cuda else Variable(_input)
                assert self.input.shape[0] == self.target.shape[0]

    def forward(self, split):
        if split == 'train':
            self.prediction = self.net(Variable(self.input))
        elif split in ['validation', 'test']:
            self.prediction = self.net(Variable(self.input, volatile=True))
            # Apply a softmax and return a segmentation map
            self.logits = self.net.apply_argmax_softmax(self.prediction)
            self.pred = self.logits.data.max(1)


    def backward(self):
        #print(self.net.apply_argmax_softmax(self.prediction), self.target)
        self.loss = self.criterion(self.prediction, self.target)
        self.loss.backward()

    def optimize_parameters(self):
        self.net.train()
        self.forward(split='train')

        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def test(self):
        self.net.eval()
        self.forward(split='test')
        self.accumulate_results()

    def validate(self):
        self.net.eval()
        self.forward(split='test')
        self.loss = self.criterion(self.prediction, self.target)
        self.accumulate_results()

    def reset_results(self):
        self.losses = []
        self.pr_lbls = []
        self.pr_probs = []
        self.gt_lbls = []

    def accumulate_results(self):
        self.losses.append(self.loss.data[0])
        self.pr_probs.append(self.pred[0].cpu().numpy())
        self.pr_lbls.append(self.pred[1].cpu().numpy())
        self.gt_lbls.append(self.target.data.cpu().numpy())

    def get_classification_stats(self):
        self.pr_lbls = np.concatenate(self.pr_lbls)
        self.gt_lbls = np.concatenate(self.gt_lbls)
        res = classification_stats(self.pr_lbls, self.gt_lbls, self.labels)
        (self.accuracy, self.f1_micro, self.precision_micro,
         self.recall_micro, self.f1_macro, self.precision_macro,
         self.recall_macro, self.confusion, self.class_accuracies,
         self.f1s, self.precisions,self.recalls) = res

        breakdown = dict(type='table',
                         colnames=['|accuracy|',' precison|',' recall|',' f1_score|'],
                         rownames=self.labels,
                         data=[self.class_accuracies, self.precisions,self.recalls, self.f1s])

        return OrderedDict([('accuracy', self.accuracy),
                            ('confusion', self.confusion),
                            ('f1', self.f1_macro),
                            ('precision', self.precision_macro),
                            ('recall', self.recall_macro),
                            ('confusion', self.confusion),
                            ('breakdown', breakdown)])

    def get_current_errors(self):
        return OrderedDict([('CE', self.loss.data[0])])

    def get_accumulated_errors(self):
        return OrderedDict([('CE', np.mean(self.losses))])

    def get_current_visuals(self):
        inp_img = util.tensor2im(self.input, 'img')
        return OrderedDict([('inp_S', inp_img)])

    def get_feature_maps(self, layer_name, upscale):
        feature_extractor = HookBasedFeatureExtractor(self.net, layer_name, upscale)
        return feature_extractor.forward(Variable(self.input))


    def save(self, epoch_label):
        self.save_network(self.net, 'S', epoch_label, self.gpu_ids)

    def set_labels(self, labels):
        self.labels = labels

    def load_network_from_path(self, network, network_filepath, strict):
        network_label = os.path.basename(network_filepath)
        epoch_label = network_label.split('_')[0]
        print('Loading the model {0} - epoch {1}'.format(network_label, epoch_label))
        network.load_state_dict(torch.load(network_filepath), strict=strict)

    def update_state(self, epoch):
        pass

    def get_fp_bp_time2(self, size=None):
        # returns the fp/bp times of the model
        if size is None:
            size = (8, 1, 192, 192)

        inp_array = Variable(torch.rand(*size)).cuda()
        out_array = Variable(torch.rand(*size)).cuda()
        fp, bp = benchmark_fp_bp_time(self.net, inp_array, out_array)

        bsize = size[0]
        return fp/float(bsize), bp/float(bsize)
