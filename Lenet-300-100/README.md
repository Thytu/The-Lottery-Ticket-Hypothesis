<br />
<div align="center">

  <h3 align="center">Lenet-300-100</h3>

  <p align="center">
    Code and result for the experiments with Lenet-300-100 on MNIST.
    <br />
  </p>


| Layer (type)     | Input Shape   | Output Shape | Param   | Tr. Param |
| ---------------- | ------------- | ------------ |-------- | --------- |
|         Linear-1 |      [1, 784] |     [1, 300] | 235,500 |     235,500
|           ReLU-2 |      [1, 300] |     [1, 300] |       0 |           0
|         Linear-3 |      [1, 300] |     [1, 100] |  30,100 |      30,100
|           ReLU-4 |      [1, 100] |     [1, 100] |       0 |           0
|         Linear-5 |      [1, 100] |      [1, 10] |   1,010 |       1,010



</div>

<br/>
<br/>


## Usage

<br/>

**To reproduce this experiment** : `python src/main.py`

Running this script will train a `Lenet-300-100` based model on `MNIST` and will prune iteratively the model.\
Each layer is puned independently by a factor $P$, defined as follow : $P=p^{1/n}$ with $p=0.2$ and $n$ the round of pruning.

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

The iteration at which early-stopping would occur of the Lenet architecture for MNIST when trained starting at various sizes (average of five trials).
<div align="center">
  <img src="./images/early_stop.png.png"/>
</div>