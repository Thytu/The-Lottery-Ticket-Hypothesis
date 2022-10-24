<br />
<div align="center">

  <h3 align="center">VGG-19</h3>

  <p align="center">
    Code and result for the experiments with VGG-19 (Simonyan & Zisserman, 2014) on CIFAR10.
    <br />
  </p>
</div>

<br/>
<br/>


## Usage

<br/>

**To reproduce this experiment** : `python src/main.py`

Running this script will train a `VGG-19` based model on `CIFAR10` and will prune iteratively the model.\
Pruning is applyied globally by a factor $P$, defined as follow : $P=p^{1/n}$ with $n$ the round of pruning.

$p=0.20$ for convolutions, $p=0$ for FC Layers.

<br/>
<br/>

## Results

<br />

Loss on the test set using Lenet (iterative pruning) as training proceeds. Each curve is the average of five trials. Labels are $Pm$—the fraction of weights remaining in the network after pruning. Error bars are the minimum and maximum of any trial.
<div align="center">
  <img src="./images/accuracies.png"/>
</div>

<br />

Test accuracy on Lenet (iterative pruning) as training proceeds. Each curve is the average of five trials. Labels are $Pm$—the fraction of weights remaining in the network after pruning. Error bars are the minimum and maximum of any trial.
<div align="center">
  <img src="./images/losses.png"/>
</div>