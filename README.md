# Sequential Graph-Based Decoding of the Surface Code using a Hybrid Graph and Recurrent Neural Network Model

This Git repo contains simplified source code for creating and testing our combined graph neural network and recurrent neural network used for decoding the surface code.

Logical accuracy | Logical failure rate
:-------------------------:|:-------------------------:
![accuracy](https://github.com/Olfj/QEC_GNN-RNN/blob/main/figures/performance_t_accuracy.png) | ![failure](https://github.com/Olfj/QEC_GNN-RNN/blob/main/figures/performance_t_failure.png)

## args.py

Contains the Args dataclass that is used to set various parameters for decoder training and inference.

## data.py

Contains the Data class that is used to generate simulated surface code data using Stim. 

## gru_decoder.py

Contains our decoder, and methods to train and evaluate it.

## mwpm.py

Contains a method that tests MWPM. Can be used as a benchmark when evaluating our decoder.

## utils.py

Contains some helper methods.

## models

Directory containing weights and biases for models trained on code distance 3, 5, and 7.

## examples

Directory containing some code examples showing how to load, train, and test our decoder.

## Usage 

```
git clone https://github.com/Olfj/QEC_GNN-RNN
cd QEC_GNN_RNN
pip3 install -r requirements.txt
python3 examples/test_nn.py
```
