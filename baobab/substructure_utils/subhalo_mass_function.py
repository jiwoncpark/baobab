import numpy as np


def lens_model_with_subhalos(substrucute_sample,pivot_mass=1e8):
    """ Return a tuple with the list of lens models and lens model kwargs for
        the subtructure.

    Parameters
    ----------
    substrucute_sample : dict
        The values of the substructure parameters sampled
    pivot_mass : float
        The pivot_mass for the subhalo mass function. Defaults to 10^8 M_sun.

    Returns
    -------
    tuple
        A tuple with first entry the lens model list for the substructure and
        the second entry the lens kwarg list for the substructure.
    """
    # First we have to calculate the overall norm of the mass function we're
    # sampling.

    # For now just put something dumb in here to make sure everything is
    # working
    lens_model_list = ['TNFW']
    lens_kwargs_list = [{'alpha_Rs':0.05, 'Rs':0.05,
        'center_x':0.1, 'center_y':-0.1,
        'r_trunc': 0.25}]
    return lens_model_list, lens_kwargs_list
