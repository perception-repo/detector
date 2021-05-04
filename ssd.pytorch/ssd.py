import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x) # 计算到vgg/lay09/relu之后，输出尺寸为[256,38,38]

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x) # 计算到vgg/lay20/relu之后，输出尺寸为[1024,19,19]
        sources.append(x)

        # apply extra layers and cache source layer outputs
        # 计算到extras/layer02/relu之后，输出尺寸为[512,10,10]
        # 计算到extras/layer04/relu之后，输出尺寸为[256,5,5]
        # 计算到extras/layer06/relu之后，输出尺寸为[256,3,3]
        # 计算到extras/layer08/relu之后，输出尺寸为[256,1,1]
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # source layers contains: [[256,38,38], [1024,19,19], [512,10,10], [256,5,5], [256,3,3], [256,1,1]]
        # apply multibox head to source layers
        # the output priors are: 38^2 × 4+19^2 × 6+10^2 × 6+5^2 × 6+3^2 × 4+1^2 × 4 = 8732
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    '''
    cfg = base['300'] = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]
    i = 3 # the initial in_channels

    The network architure is as following:

    layer 01: [nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)], size: [64,300,300]
    layer 02: [nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)], size: [64,300,300]
    layer 03: [nn.MaxPool2d(kernel_size=2, stride=2)], size: [64, 150, 150]
    layer 04: [nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)], size: [128,150,150]
    layer 05: [nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)], size: [128,150,150]
    layer 06: [nn.MaxPool2d(kernel_size=2, stride=2)], size: [128, 75, 75]
    layer 07: [nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)], size: [256,75,75]
    layer 08: [nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)], size: [256,75,75]
    layer 09: [nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)], size: [256,75,75]
    layer 10: [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)], size: [256, 38, 38] # 剩余数据不足以构成2x2时，计算剩余值的最大值

    layer 11: [nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True)], size: [512,38,38]
    layer 12: [nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True)], size: [512,38,38]
    layer 13: [nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True)], size: [512,38,38]
    layer 14: [nn.MaxPool2d(kernel_size=2, stride=2)], size: [512, 19, 19]

    layer 15: [nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True)], size: [512,19,19]
    layer 16: [nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True)], size: [512,19,19]
    layer 17: [nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True)], size: [512,19,19]

    layer 18: [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)], size: [512,19,19]
    layer 19: [nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6), nn.ReLU(inplace=True)], size: [1024, 19, 19]
    layer 20: [nn.Conv2d(1024, 1024, kernel_size=1), nn.ReLU(inplace=True)], size: [1024, 19, 19]
    '''
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    '''
    cfg = [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]
    i = 1024 # the initial in_channels

    layer 01: [nn.Conv2d(1024, 256, kernel_size=1)]
    layer 02: [nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)], size: [?,0.5,0.5]
    layer 03: [nn.Conv2d(512, 128, kernel_size=1)]
    layer 04: [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)], size: [?,0.5,0.5]
    layer 05: [nn.Conv2d(256, 128, kernel_size=1)]
    layer 06: [nn.Conv2d(128, 256, kernel_size=3)]
    layer 07: [nn.Conv2d(256, 128, kernel_size=1)]
    layer 08: [nn.Conv2d(128, 256, kernel_size=3)]
    '''
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    '''
    cfg = [4, 6, 6, 6, 4, 4]
    num_classes = 21

    loc_layer:
        layer 01: [nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1)], size: [38,38]
        layer 02: [nn.Conv2d(1024, 6 * 4, kernel_size=3, padding=1)], size: [19, 19]
        layer 03: [nn.Conv2d(512, 6 * 4, kernel_size=3, padding=1)], size: [?, ?]
        layer 04: [nn.Conv2d(256, 6 * 4, kernel_size=3, padding=1)], size: [?, ?]
        layer 05: [nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1)], size: [?, ?]
        layer 06: [nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1)], size: [?, ?]

    conf_layers:
        layer 01: [nn.Conv2d(256, 4 * 21, kernel_size=3, padding=1)], size: [38,38]
        layer 02: [nn.Conv2d(1024, 4 * 21, kernel_size=3, padding=1)], size: [19, 19]
        layer 03: [nn.Conv2d(512, 6 * 21, kernel_size=3, padding=1)], size: [?, ?]
        layer 04: [nn.Conv2d(256, 6 * 21, kernel_size=3, padding=1)], size: [?, ?]
        layer 05: [nn.Conv2d(256, 4 * 21, kernel_size=3, padding=1)], size: [?, ?]
        layer 06: [nn.Conv2d(256, 4 * 21, kernel_size=3, padding=1)], size: [?, ?]
    '''
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes)
