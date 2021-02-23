# Hadamard Codebook based Deep Hashing

This resposity is the pytorch implementation of our paper HCDH: Hadamard Codebook based Deep Hashing ([link](https://arxiv.org/pdf/1910.09182.pdf)).

![image-20210217173236178](https://tva1.sinaimg.cn/large/008eGmZEly1gnqnxabkeoj312m0czq65.jpg)

## Requirements
- Python 3.6
- PyTorch == 1.4.0

## Datasets
The data list of datasets **CIFAR-10**、**NUS-WIDE** and **ImageNet** are provides in [Baidu Yun](https://pan.baidu.com/s/1Kxnmo7b07OL_NKtsCRi2DA) (code: tkrb).

## Usage
**Train/Test**
``` bash
python main.py --gpus 0 --dataset cifar_s1 --feat-dim 32 --prefix AlexNet_32bit
```

## Citation
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follows.

```BibTeX
@article{Chen2019HadamardCB,
  title={Hadamard Codebook Based Deep Hashing},
  author={S. Chen and Liujuan Cao and Mingbao Lin and Yan Wang and Xiaoshuai Sun and Chenglin Wu and Jingfei Qiu and Rongrong Ji},
  journal={ArXiv},
  year={2019},
  volume={abs/1910.09182}
}
```
