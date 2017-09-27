## 图像分割SegNet
    ### 本人博客：http://blog.csdn.net/sinat_26917383/article/details/77825764

### 相关参考
- github链接：https://github.com/chainer/chainercv
- 官方文档链接：http://chainercv.readthedocs.io/en/stable/index.html
- 预训练模型下载页面：https://github.com/yuyu2172/share-weights/releases/
- SegNet参考的caffe实现及预训练模型：http://mi.eng.cam.ac.uk/projects/segnet/tutorial.html


由chainercv自带的预训练模型


### 采用的数据集为：camvid

- 数据集类别camvid_label_names：'Sky',  'Building',  'Pole',  'Road',  'Pavement',  'Tree',  'SignSymbol',  'Fence',  'Car',  'Pedestrian',  'Bicyclist'
- 不同类别的颜色camvid_label_colors：(128, 128, 128),  (128, 0, 0),  (192, 192, 128),  (128, 64, 128),  (60, 40, 222),  (128, 128, 0),  (192, 128, 128),  (64, 64, 128),  (64, 0, 128),  (64, 64, 0),  (0, 128, 192)


### 其中需要注意的是：

- (1)'pip install chainercv'好像没有load进去vis_semantic_segmentation模块，所以我的做法是从github中加到：/usr/local/lib/python3.5/dist-packages/chainercv/visualizations目录下（github该模块链接：https://github.com/chainer/chainercv/tree/master/chainercv/visualizations）
- (2)读图的时候，注意最好使用chainercv自带的读入函数utils.read_image


### 2、自己训练segnet模型
--------------

其他的，如果你要自己训练segnet模型，请参考[该页面](https://github.com/chainer/chainercv/tree/master/examples/segnet)

First, move to this directory (i.e., examples/segnet) and run:

```
python train.py [--gpu <gpu>]
```
一个使用的demo：

```
wget https://raw.githubusercontent.com/alexgkendall/SegNet-Tutorial/master/CamVid/test/0001TP_008550.png
python demo.py [--gpu <gpu>] [--pretrained_model <model_path>] 0001TP_008550.png
```
模型评估的函数：

```
python eval_camvid.py [--gpu <gpu>] [--pretrained_model <model_path>] [--batchsize <batchsize>]
```
这里有一个已经训练好的模型，可以做个案例，预训练模型的下载链接为：https://www.dropbox.com/s/exas66necaqbxyw/model_iteration-16000
评估的结果展示：
![这里写图片描述](http://img.blog.csdn.net/20170927112043613?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2luYXRfMjY5MTczODM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

官方自带的一个预训练模型后的模型使用demo：

```
import argparse
import matplotlib.pyplot as plot

import chainer

from chainercv.datasets import camvid_label_colors
from chainercv.datasets import camvid_label_names
from chainercv.links import SegNetBasic
from chainercv import utils
from chainercv.visualizations import vis_image
from chainercv.visualizations import vis_semantic_segmentation


def main():
    chainer.config.train = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained_model', default='camvid')
    parser.add_argument('image')
    args = parser.parse_args()

    model = SegNetBasic(
        n_class=len(camvid_label_names),
        pretrained_model=args.pretrained_model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    img = utils.read_image(args.image, color=True)
    labels = model.predict([img])
    label = labels[0]

    fig = plot.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    vis_image(img, ax=ax1)
    ax2 = fig.add_subplot(1, 2, 2)
    vis_semantic_segmentation(
        label, camvid_label_names, camvid_label_colors, ax=ax2)
    plot.show()


if __name__ == '__main__':
    main()
```
