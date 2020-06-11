# TransdeepOCSVM: A Deep Hybrid Model For Anomaly Detection 

This repository contains the PyTorch code of the paper "TransdeepOCSVM: A Deep Hybrid Model For Anomaly Detection", the repository is largely based on the official code for the paper "GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training" [[1]](#reference)


Some example commands to replicate the paper's results for MNIST and CIFAR10  datasets are presented below:

``` shell
# MNIST
python3 train.py --dataset mnist --isize 32 --nc 1 --niter 15 --abnormal_class 4 --manualseed 0

# CIFAR
python3 train.py --dataset cifar10 --isize 32 --niter 15 --abnormal_class plane --manualseed 0
```

## 6. Reference
[1]  Akcay S., Atapour-Abarghouei A., Breckon T.P. (2019) GANomaly: Semi-supervised Anomaly Detection via Adversarial Training. In: Jawahar C., Li H., Mori G., Schindler K. (eds) Computer Vision – ACCV 2018. ACCV 2018. Lecture Notes in Computer Science, vol 11363. Springer, Cham
