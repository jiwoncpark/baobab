# baobab

Training data generator for hierarchical inference with Bayesian neural networks

### Installation

0. You'll need a Fortran compiler, which you can get on a debian system by running
```shell
sudo apt-get install gfortran
```

1. Virtual environments are strongly recommended, to prevent dependencies with conflicting versions. Create a conda virtual environment and activate it.
```shell
conda create -n baobab python=3.6 -y
conda activate baobab
```

2. Now do one of the following. 

## Option 2(a): clone the repo (please do this if you'd like to contribute to the development).
```
git clone https://github.com/jiwoncpark/baobab.git
cd baobab
pip install -e .
```

## Option 2(b): pip install the release version (only recommended if you're a user).
```
pip install baobab
```

3. (Optional) To run the notebooks, add the Jupyter kernel.
```shell
python -m ipykernel install --user --name baobab --display-name "Python (baobab)"
```

### Usage

1. Choose your favorite config file among the templates in the `configs` directory and *copy* it to a directory of your choice, e.g.
```shell
mkdir my_config_library
cp baobab/configs/tdlmc_diagonal_config.py my_config_library/my_config.py
```

2. Customize it! You might want to change the `name` field first with something recognizable. Pay special attention to the `components` field, which determines which components of the lensed system (e.g. lens light, AGN light) become sampled from relevant priors and rendered in the image.

2. Generate the training set, e.g. continuing with the example in #1,
```shell
generate my_config.py
```

Although the `n_data` (size of training set) value is specified in the config file, you may choose to override it in the command line, as in
```shell
generate my_config.py 100
```

Please message @jiwoncpark with any questions.

There is an ongoing [document](https://www.overleaf.com/read/pswdqwttjbjr) that details our BNN prior choice, written and maintained by Ji Won.