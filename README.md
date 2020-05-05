# Boltzmann Machine
A stochastic neural network that learns to reconstruct the input's probability distribution either in supervised or unsupervised settings

Student Project by: Michael Blesel, Oliver Pola

In Seminar: [Neural Networks Seminar 2020](https://www.inf.uni-hamburg.de/en/inst/ab/wtm/teaching.html), Knowledge Technology Research Group, Uni Hamburg

References:

Ackley, Hinton, Sejnowski
[A Learning Algorithm for Boltzmann Machines](https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog0901_7)
Cognitive Science 9
pp. 147-169
January 1985

---

## Setup
We'll use pipenv as a virtual environment. To set everything up and run somefile.py, start from here where the README.md is and do:

```
pip install pipenv
pipenv sync
pipenv shell
python somefile.py
```

The required Python modules are listed in the provided `Pipfile` and installed via `pipenv install` command. To use the exact versions specified in `Pipfile.lock`, use `pipenv sync`. If you like to skip the virtual environment, you could also install all modules listed in  `Pipfile` via `pip install ...`.
