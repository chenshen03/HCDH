# Hadamard Codebook based Deep Hashing

This resposity is the pytorch implementation of our paper HCDH: Hadamard Codebook based Deep Hashing ([link](https://arxiv.org/pdf/1910.09182.pdf)).


## Dependency
- Python 3.6
- PyTorch == 1.4.0

## Datasets
- CIFAR
- NUS-WIDE
- ImageNet

## Usage
**Train/Test**
``` bash
python main.py --gpus 0 --dataset cifar_s1 --feat-dim 32 --prefix AlexNet_32bit
```

## Citations
If our paper helps your research, please cite it in your publications:
```
@article{Chen2019HadamardCB,
  title={Hadamard Codebook Based Deep Hashing},
  author={S. Chen and Liujuan Cao and Mingbao Lin and Yan Wang and Xiaoshuai Sun and Chenglin Wu and Jingfei Qiu and Rongrong Ji},
  journal={ArXiv},
  year={2019},
  volume={abs/1910.09182}
}
```
