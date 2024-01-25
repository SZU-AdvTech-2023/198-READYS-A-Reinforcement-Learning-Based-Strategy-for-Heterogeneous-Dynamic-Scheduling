This is a Pytorch implementation of the paper: [READYS: A Reinforcement Learning Based Strategy for Heterogeneous Dynamic Scheduling](https://hal.inria.fr/hal-03313229/)


## Requirements

  * Python 3.11.4
  * For the other packages, please refer to the requirements.txt.
  * Some packages (PyTorch, torch-geometric) may have to be installed separately.


## Usage

The main script is `train.py` . You can reproduce the results by:
```
pip install -r requirements.txt
python train.py
```

## Options

### DAG types
It is possible to train on three types of DAG: **Cholesky**, **QR**, **LU**, like in the following examples:

```
python train.py -env_type chol
python train.py -env_type QR
python train.py -env_type LU
```
### Number of tiles *T*

The parameter **n** controls the number of tiles *T*. As the graphs grow rapidly with the number of tiles, the training can become slow from 10 onwards.
```
python train.py -env_type QR -n 4
```

For other hyper-parameters concerning the RL algorithm or training, please look at the parser arguments at the top of the train file.


