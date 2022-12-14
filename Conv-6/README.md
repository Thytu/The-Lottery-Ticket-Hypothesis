<br />
<div align="center">

  <h3 align="center">Conv6</h3>

  <p align="center">
    Code and result for the experiments with Conv6, variants of VGG (Simonyan & Zisserman, 2014) on CIFAR10.
    <br />
  </p>
</div>

<br/>
<br/>


## Usage

<br/>

**To reproduce this experiment** : `python src/main.py`

Running this script will train a `Conv6` based model on `CIFAR10` and will prune iteratively the model.\
Each layer is puned independently by a factor $P$, defined as follow : $P=p^{1/n}$ with $n$ the round of pruning.

$p=0.15$ for convolutions, $p=0.2$ for FC Layers and $p=0.1$ for the output layer.

<br/>
<br/>

## Results

<br />

Loss on the test set using Conv-6 (iterative pruning) as training proceeds. Each curve is the average of five trials. Labels are $Pm$—the fraction of weights remaining in the network after pruning. Error bars are the minimum and maximum of any trial.
<div align="center">
  <img src="./images/accuracies.png"/>
</div>

<br />

Test accuracy on Conv-6 (iterative pruning) as training proceeds. Each curve is the average of five trials. Labels are $Pm$—the fraction of weights remaining in the network after pruning. Error bars are the minimum and maximum of any trial.
<div align="center">
  <img src="./images/losses.png"/>
</div>

<br />

The iteration at which early-stopping would occur of the Conv-6 architecture for CIFAR10 when trained starting at various sizes (average of five trials).
<div align="center">
  <img src="./images/early_stop.png.png"/>
</div>
