=====
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


Please message @jiwoncpark with any questions.