# qp_formulations

## Install

First you need to clone the repository
```
mkdir project_qp_formulations
cd project_qp_formulations
git clone  https://gitlab.laas.fr/nvilla/qp_formulations
```

Then to create a virtualenv

```
virtualenv venv
source ./venv/bin/activate
```

Install numpy:
```
pip install numpy simpy scipy
```

## Starting the tests

```
export PYTHONPATH=$PWD
python3 ./tests/test_body.py
```
should give:
```
..
----------------------------------------------------------------------
Ran 2 tests in 0.256s

OK
```

Testing the dynamics gives:
```
python3 ./tests/test_dynamics.py 
...
----------------------------------------------------------------------
Ran 4 tests in 0.298s

OK
```

Testing the restrictions gives:
```
python3 ./tests/test_restrictions.py 
.....
----------------------------------------------------------------------
Ran 5 tests in 0.003s

OK
```

