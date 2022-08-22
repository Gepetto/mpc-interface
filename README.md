[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# MPC interface


This package provides a structure to formulate QP based MPC problems with any linear dynamics, cost functions and constraint.
All required QP matrices are generated automatically with the structure considered in [qpsolvers](https://scaron.info/doc/qpsolvers/).

This repository is mainly based on python, the c++ part is a work in progress which does not work yet.


## Dependancies

`mpc_interface` depends on standard python libraries and qpsolvers + osqp for the examples.

### Standard python packages

[`pip3 install numpy`](https://numpy.org/install/)
[`pip3 install sympy`](https://pypi.org/project/sympy/)
[`pip3 install matplotlib`](https://matplotlib.org/stable/users/installing/index.html)
[`pip3 install scipy`](https://scipy.org/install/)


### External packages

[`pip3 install qpsolvers`](https://pypi.org/project/qpsolvers/)
[`pip3 install osqp`](https://osqp.org/docs/get_started/python.html)

## Usage

A better explanation is comming soon.
For now, an embryonary explanation of the repository classes is provided in [`biped_formulation.py`](https://github.com/Gepetto/mpc-interface/blob/main/python/use_examples/simple_functional_example/biped_formulation.py)

### Tests

Run all the tests with:

```bash
python3 -m unittest discover python/tests
```

### Check the examples

Available examples are in the folder [`python/use_examples`](https://github.com/Gepetto/mpc-interface/tree/main/python/use_examples)

Run the starting example with

```bash
ipython python/use_examples/simple_functional_example/biped_mpc_loop.py
```
