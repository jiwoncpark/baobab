Baobab
======

The Baobab package can generate images of strongly-lensed systems, given some configurable prior distributions over the parameters of the lens and light profiles as well as configurable assumptions about the instrument and observation conditions. It supports prior distributions ranging from artificially simple to empirical.

A major use case for Baobab is the generation of training and test sets for hierarchical inference using Bayesian neural networks (BNNs). The idea is that Baobab will generate the training and test sets using different priors. A BNN trained on the training dataset learns not only the parameters of individual lens systems but also, implicitly, the hyperparameters describing the training set population (the training prior). Such hierarchical inference is crucial in scenarios where the training and test priors are different, so that techniques such as importance weighting can be employed to bridge the gap in the BNN response.

**Contents:**

.. toctree::
   :maxdepth: 2

   installation
   usage
   configs
   bnn_prior
   generate
   to_hdf5


