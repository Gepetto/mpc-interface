[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# MPC interface

[![Pipeline status](https://gitlab.laas.fr/gepetto/mpc-interface/badges/master/pipeline.svg)](https://gitlab.laas.fr/gepetto/mpc-interface/commits/master)
[![Coverage report](https://gitlab.laas.fr/gepetto/mpc-interface/badges/master/coverage.svg?job=doc-coverage)](https://gepettoweb.laas.fr/doc/gepetto/mpc-interface/master/coverage/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/gepetto/mpc-interface/master.svg)](https://results.pre-commit.ci/latest/github/gepetto/mpc-interface)

This package provides a structure to formulate QP based MPC problems with any linear dynamics, cost functions and constraint.
All required QP matrices are generated automatically with the structure considered in [qpsolvers](https://scaron.info/doc/qpsolvers/).

This repository is mainly based on python, the c++ part is a work in progress which does not work yet.


## Dependancies

`mpc_interface` depends on the following python packages

[`numpy`](https://numpy.org/install/)
[`sympy`](https://pypi.org/project/sympy/)
[`matplotlib`](https://matplotlib.org/stable/users/installing/index.html)
[`scipy`](https://scipy.org/install/)

that can be installed as follows:

```bash
pip3 install numpy sympy matplotlib scipy
```

The examples require additionally:

[`qpsolvers`](https://pypi.org/project/qpsolvers/)
[`osqp`](https://osqp.org/docs/get_started/python.html)

```bash
pip3 install qpsolvers osqp
```

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
