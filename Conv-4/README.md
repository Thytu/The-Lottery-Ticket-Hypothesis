<br />
<div align="center">

  <h3 align="center">Conv4</h3>

  <p align="center">
    Code and result for the experiments with Conv4, variants of VGG (Simonyan & Zisserman, 2014) on CIFAR10.
    <br />
  </p>

| Layer (type)  | Input Shape          | Output Shape      | Param     | Tr. Param |
| ------------- | -------------------- | ----------------- | --------- | --------- |
|         Conv2d-1  |   [1, 3, 32, 32] |   [1, 64, 32, 32] |     1,792 |       1,792
|           ReLU-2  |  [1, 64, 32, 32] |   [1, 64, 32, 32] |         0 |           0
|         Conv2d-3  |  [1, 64, 32, 32] |   [1, 64, 32, 32] |    36,928 |      36,928
|      MaxPool2d-4  |  [1, 64, 32, 32] |   [1, 64, 16, 16] |         0 |           0
|           ReLU-5  |  [1, 64, 16, 16] |   [1, 64, 16, 16] |         0 |           0
|         Conv2d-6  |  [1, 64, 16, 16] |  [1, 128, 16, 16] |    73,856 |      73,856
|           ReLU-7  | [1, 128, 16, 16] |  [1, 128, 16, 16] |         0 |           0
|         Conv2d-8  | [1, 128, 16, 16] |  [1, 128, 16, 16] |   147,584 |     147,584
|      MaxPool2d-9  | [1, 128, 16, 16] |    [1, 128, 8, 8] |         0 |           0
|          ReLU-10  |   [1, 128, 8, 8] |    [1, 128, 8, 8] |         0 |           0
|        Linear-11  |        [1, 8192] |          [1, 256] | 2,097,408 |   2,097,408
|          ReLU-12  |         [1, 256] |          [1, 256] |         0 |           0
|        Linear-13  |         [1, 256] |          [1, 256] |    65,792 |      65,792
|          ReLU-14  |         [1, 256] |          [1, 256] |         0 |           0
|        Linear-15  |         [1, 256] |           [1, 10] |     2,570 |       2,570

</div>

<br/>
<br/>


## Usage

<br/>

**To reproduce this experiment** : `python src/main.py`

Running this script will train a `Conv4` based model on `CIFAR10` and will prune iteratively the model.\
Each layer is puned independently by a factor $P$, defined as follow : $P=p^{1/n}$ with $n$ the round of pruning.

$p=0.1$ for convolutions, $p=0.2$ for FC Layers and $p=0.1$ for the output layer.

<br/>
<br/>

## Results

<br />

Loss on the test set using Conv-4 (iterative pruning) as training proceeds. Each curve is the average of five trials. Labels are $Pm$—the fraction of weights remaining in the network after pruning. Error bars are the minimum and maximum of any trial.
<div align="center">
  <img src="./images/accuracies.png"/>
</div>

<br />

Test accuracy on Conv-4 (iterative pruning) as training proceeds. Each curve is the average of five trials. Labels are $Pm$—the fraction of weights remaining in the network after pruning. Error bars are the minimum and maximum of any trial.
<div align="center">
  <img src="./images/losses.png"/>
</div>

<br />

The iteration at which early-stopping would occur of the Conv-4 architecture for CIFAR10 when trained starting at various sizes (average of five trials).
<div align="center">
  <img src="./images/early_stop.png.png"/>
</div>
