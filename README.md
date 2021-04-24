# Detectors

## 1. Components requiring more attention

### 1.1 How to data augmentation

数据增强过程中需要对图像数据和真值数据进行分别处理，所以通常在`Dataset`的`__getitem()__`函数返回值处，将图像数据作为一个独立的参数返回，然后再将其它真值数据作为打包为一个`tuple`一起返回。这样便于在后面的数据增强过程中，对图像数据和真值数据进行差异化处理。

在对图像进行裁剪操作时，注意同时更新`bbox`的信息。

对图像进行放缩操作前，要先将`bbox`及`segmentation`的坐标信息转换为相对坐标。

自定义数据增强方法的样例代码见[链接](https://github.com/perception-repo/detector/blob/c7dc09f44bed6400414fb1e29a47107bb90a5ce1/ssd.pytorch/utils/augmentations.py#L400).

### 1.2 How to design `model`

### 1.3 How to match `targets` with `predictions` and calculate `loss`

在训练的过程中，不需要将网络输出的`bbox`回归量进行解码，而是将`gt`和`prior`匹配后将其编码为和预测结果一致的形式。

首先将`priors`或`feature map`和`targets`进行匹配，得到和`predictions`形式一致的真值信息。然后将`predictions`进行计算调整。最后基于二者的结果计算损失函数。

对于不同结构的网络，将`proposals`、`bbox regression`和`bbox confidence`整合的方法都不相同，所以私以为在**标准**的实现中，应该将`image`和`target`一起作为`model`的输入（测试阶段`targets=None`）。若有有效的真值信息输入，网络则将根据真值信息，将真值处理为可以和自己的预测对对齐的形式。同时也将自己的预测转化为有效的`bbox`数据形式。对于和模型耦合的部分，应该放到模型文件里面。

算法流程如下：

```python
def generate_anchors(cfg['model']):
    proposals = torch.Tensor((0, 4), required_grad=False)
    return proposals

def assign_targets_to_anchors(anchors, targets):
    for anchors_per_image, targets_per_image in zip(anchors, targets):
        gt_boxes = targets_per_image["boxes"]
        match_quality_matrix = box_similarity(gt_boxes, anchors_per_image)
        # for each proposal, if the match_quality(iou) > threshold, add target_label to the proposal
```

## 2. Training Techniques

