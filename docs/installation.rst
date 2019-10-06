============
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
$pip install -e .

**Option 2(b):** pip install the release version (only recommended if you're a user).

::

$pip install baobab


3. (Optional) To run the notebooks, add the Jupyter kernel.

::

$python -m ipykernel install --user --name baobab --display-name "Python (baobab)"