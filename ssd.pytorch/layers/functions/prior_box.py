from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorBox(object):
    """
    Compute priorbox coordinates in center-offset form for each source feature map.
    所谓priorbox实际上就是网格中每一个cell推荐的box
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps): # 存放的是feature map的尺寸: 38, 19, 10, 5, 3, 1
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k] # steps=[8,16,32,64,100,300]. f_k大约为feature map的尺寸
                # unit center x,y
                # 求得center的坐标, 浮点类型. 实际上, 这里也可以直接使用整数类型的 `f`, 计算上没太大差别
                # 这里一定要特别注意 i,j 和cx, cy的对应关系, 因为cy对应的是行, 所以应该零cy与i对应.
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                # 'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
                # 综上, 每个卷积特征图谱上每个像素点最终产生的 box 数量要么为4, 要么为6, 根据不同情况可自行修改
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
