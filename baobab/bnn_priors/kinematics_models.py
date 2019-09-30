import numpy as np

def velocity_dispersion_function_CPV2007(vel_disp_grid):
    """Evaluate the velocity dispersion function from the fit on SDSS DR6
    by [1]_ on a provided grid and normalizes the result to unity, so it can 
    be used as a PMF from which to draw the velocity dispersion.

    Parameters
    ----------
    vel_disp_grid : array-like
        a grid of velocity dispersion values in km/s

    Returns
    -------
    array-like, same shape as `vel_disp_grid`
        the velocity dispersion function evaluated at `vel_disp_grid`
        and normalized to unity

    Note
    ----
    The returned array is normalized to unity and we treat it as a PMF from which to sample
    the velocity dispersion. We also use the exact fit values also used in LensPop ([2]_).

    References
    ----------
    .. [1] Choi, Yun-Young, Changbom Park, and Michael S. Vogeley. 
    "Internal and collective properties of galaxies in the Sloan Digital Sky Survey." 
    The Astrophysical Journal 658.2 (2007): 884.

    .. [2] Collett, Thomas E. 
    "The population of galaxy–galaxy strong lenses in forthcoming optical imaging surveys." 
    The Astrophysical Journal 811.1 (2015): 20.
    
    """
    #h = true_H0/100.0
    #phi_star = 8.0*1.e-3
    sig_star = 161.0
    alpha = 2.32
    beta = 2.67
    #beta_over_gamma = 2.43827086163172 # beta/gamma(alpha/beta) for alpha=2.32, beta=2.67
    dn = (vel_disp_grid/sig_star)**alpha
    dn *= np.exp(-(vel_disp_grid/sig_star)**beta)
    #dn *= beta_over_gamma
    dn *= 1.0/vel_disp_grid
    #dn *= phi_star * h**3.0
    return dn