# Natural Parameter Networks

Natural Parameter Networks are a sampling/variational inference free way for Bayesian Deep Learning, based on the paper [here]{https://arxiv.org/pdf/1611.00448.pdf}.

This repository implements the Gaussian variant of NPN for classification and extends this idea to recurrent architectures.

*To run MNIST example*
```
python main-mnist.py
```

*To run LM example*
```
python main-lm.py
```

This has been done for course research project for 10-708 Probablistic Graphical Models at Carnegie Mellon University.

## TODO
* Implementation of NPRN
* Gaussian NPN for regression
* Implement other variants
