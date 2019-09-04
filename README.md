# baobab

Training data generator for hierarchical inference with Bayesian neural networks

### Installation

1. Create a conda virtual environment and activate it.
```shell
conda create -n baobab python=3.6 -y
conda activate baobab
```

2. Install some dependencies.
```shell
pip install -r requirements.txt
```

3. Install `fastell4py`.
```shell
git clone https://github.com/sibirrer/fastell4py <DESIRED DESTINATION>
cd <DESIRED DESTINATION>/fastell4py
python setup.py install --user
```
Note: this requires a Fortran compiler, e.g. on a debian system
```shell
sudo apt-get install gfortran
```

4. (Optional) To run the notebooks, add the Jupyter kernel.
```shell
python -m ipykernel install --user --name baobab --display-name "Python (baobab)"
```

### Usage

1. *Copy* your favorite config file in the `configs` directory, e.g. `configs/tdlmc_config.py`, and customize it. You might want to change the `name` field first with something recognizable. Pay special attention to the `components` field, which determines which components of the lensed system (e.g. lens light, AGN light) become sampled from relevant priors and rendered in the image.

2. Run
```shell
cd baobab
python generate.py <path to your config file>
# e.g. python generate.py configs/tdlmc_config.py
```

Please message @jiwoncpark with any questions.

There is an ongoing [document](https://www.overleaf.com/read/pswdqwttjbjr) that details our BNN prior choice, written and maintained by Ji Won.