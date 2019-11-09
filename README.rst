======
Baobab
======

.. image:: https://travis-ci.com/jiwoncpark/baobab.svg?branch=master
    :target: https://travis-ci.org/jiwoncpark/baobab

.. image:: https://readthedocs.org/projects/pybaobab/badge/?version=latest
        :target: https://pybaobab.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://coveralls.io/repos/github/jiwoncpark/baobab/badge.svg?branch=master
        :target: https://coveralls.io/github/jiwoncpark/baobab?branch=master


Training data generator for hierarchically modeling strong lenses with Bayesian neural networks

The ``baobab`` package can generate images of strongly-lensed systems, given some configurable prior distributions over the parameters of the lens and light profiles as well as configurable assumptions about the instrument and observation conditions. It supports prior distributions ranging from artificially simple to empirical.

A major use case for ``baobab`` is the generation of training and test sets for hierarchical inference using Bayesian neural networks (BNNs). The idea is that Baobab will generate the training and test sets using different priors. A BNN trained on the training dataset learns not only the parameters of individual lens systems but also, implicitly, the hyperparameters describing the training set population (the training prior). Such hierarchical inference is crucial in scenarios where the training and test priors are different, so that techniques such as importance weighting can be employed to bridge the gap in the BNN response.

Installation
============

0. You'll need a Fortran compiler and Fortran-compiled `fastell4py`, which you can get on a debian system by running

::

$sudo apt-get install gfortran
$git clone https://github.com/sibirrer/fastell4py.git <desired location>
$cd <desired location>
$python setup.py install --user

1. Virtual environments are strongly recommended, to prevent dependencies with conflicting versions. Create a conda virtual environment and activate it:

::

$conda create -n baobab python=3.6 -y
$conda activate baobab

2. Now do one of the following. 

**Option 2(a):** clone the repo (please do this if you'd like to contribute to the development).

::

$git clone https://github.com/jiwoncpark/baobab.git
$cd baobab
$pip install -e . -r requirements.txt

**Option 2(b):** pip install the release version (only recommended if you're a user).

::

$pip install baobab


3. (Optional) To run the notebooks, add the Jupyter kernel.

::

$python -m ipykernel install --user --name baobab --display-name "Python (baobab)"

Usage
=====

1. Choose your favorite config file among the templates in the `configs` directory and *copy* it to a directory of your choice, e.g.

::

$mkdir my_config_collection
$cp baobab/configs/tdlmc_diagonal_config.py my_config_collection/my_config.py


2. Customize it! You might want to change the `name` field first with something recognizable. Pay special attention to the `components` field, which determines which components of the lensed system (e.g. lens light, AGN light) become sampled from relevant priors and rendered in the image.

3. Generate the training set, e.g. continuing with the example in #1,

::

$generate my_config_collection/my_config.py

Although the `n_data` (size of training set) value is specified in the config file, you may choose to override it in the command line, as in

::

$generate my_config_collection/my_config.py 100

Feedback
========

Please message @jiwoncpark with any questions.

There is an ongoing `document <https://www.overleaf.com/read/pswdqwttjbjr>`_ that details our BNN prior choice, written and maintained by Ji Won.

Attribution
===========

``baobab`` heavily uses ``lenstronomy``, a multi-purpose package for modeling and simulating strongly-lensed systems (see `source <https://github.com/sibirrer/lenstronomy>`_). When you use ``baobab`` for your project, please cite ``lenstronomy`` with `Birrer & Amara 2018 <https://arxiv.org/abs/1803.09746v1>`_ as well as Park et al. 2019 (in prep).