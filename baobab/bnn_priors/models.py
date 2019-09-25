import numpy as np
from scipy.special import gamma
import astropy.units as u

def velocity_dispersion_function_CPV2007(vel_disp_grid):
    """Evaluate the velocity dispersion function from the fit on SDSS DR6
    by [1]_ on a provided grid.

    Parameters
    ----------
    vel_disp_grid : array-like
        a grid of velocity dispersion values in km/s

    Returns
    -------
    array-like, same shape as `vel_disp_grid`
        the unnormalized velocity dispersion function

    Note
    ----
    The returned function is unnormalized. We use the exact fit values also used in LensPop ([2]_).

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
    dn = (vel_disp_grid/sig_star)**alpha
    dn *= np.exp(-(vel_disp_grid/sig_star)**beta)
    dn *= beta/gamma(alpha/beta)
    dn *= 1.0/vel_disp_grid
    #dn *= phi_star * h**3.0
    return dn

def luminosity_from_faber_jackson(vel_disp, slope=2.0, intercept=5.4):
    """Evaluate the V-band luminosity L_V expected from the Faber-Jackson (FJ) relation
    for a given velocity dispersion

    Parameters
    ----------
    vel_disp : float
        the velocity dispersion in km/s
    slope : float
        slope of the log(L_V/L_solar) vs. log(vel_disp) relation
    intercept : float
        intercept of the log(L_V/L_solar) vs. log(vel_disp) relation

    Returns
    -------
    float
        log(L_V/L_solar)

    Note
    ----
    The FJ relation is a projection of the Fundamental Plane (FP) relation.
    The default values of slope and intercept are those expected for ETGs. See Fig 7 of [1]_.
    Values binned by magnitudes are available in [2]_.
    V-band has rest-frame wavelength range 480nm ~ 650nm

    References
    ----------
    .. [1] D’Onofrio, Mauro, et al. 
    "On the Origin of the Fundamental Plane and Faber–Jackson Relations: Implications for the Star Formation Problem." 
    The Astrophysical Journal 838.2 (2017): 163.

    .. [2] Nigoche-Netro, A., et al. 
    "The Faber-Jackson relation for early-type galaxies: dependence on the magnitude range." 
    Astronomy & Astrophysics 516 (2010): A96.

    """
    log_L_V = slope*np.log10(vel_disp) + intercept
    return log_L_V

def size_from_fundamental_plane(vel_disp, m_V, a=1.4335, b=0.3150, c=-8.8979):
    """Evaluate the size expected from the Fundamental Plane (FP) relation
    for a given velocity dispersion and V-band apparent magnitude

    Parameters
    ----------
    vel_disp : float
        the velocity dispersion in km/s
    m_V : float
        the apparent V-band magnitude
    slope : float
        slope of the log(L_V/L_solar) vs. log(vel_disp) relation
    intercept : float
        intercept of the log(L_V/L_solar) vs. log(vel_disp) relation

    Returns
    -------
    float
        the effective radius in kpc

    Note
    ----
    The default values of slope and intercept are taken from the r-band orthogonal fit
    on SDSS DR4. See Table 2 of [1]_.
    V-band has rest-frame wavelength range 480nm ~ 650nm.

    References
    ----------
    .. [1] Hyde, Joseph B., and Mariangela Bernardi. 
    "The luminosity and stellar mass Fundamental Plane of early-type galaxies." 
    Monthly Notices of the Royal Astronomical Society 396.2 (2009): 1171-1185.

    """
    log_R_eff = a*np.log10(vel_disp) + b*m_V + c # in kpc
    R_eff = 10**log_R_eff
    return R_eff

def gamma_from_size_relation(R_eff, slope=-0.41, intercept=0.39, delta_slope=0.12, delta_intercept=0.10, intrinsic_scatter=0.14):
    """Evaluate the power-law slope of the mass profile using the empirical correlation derived 
    from the SLACS lens galaxy sample with a given effective radius

    Parameters
    ----------
    R_eff : float
        the effective radius in kpc

    Note
    ----
    See Table 4 of [1]_ for the default fit values used.

    References
    ----------
    .. [1] Auger, M. W., et al. 
    "The Sloan Lens ACS Survey. X. Stellar, dynamical, and total mass correlations of massive early-type galaxies." 
    The Astrophysical Journal 724.1 (2010): 511.

    """
    log_R_eff = np.log10(R_eff)
    gamma_minus_2 = log_R_eff*slope + intercept
    gamma = gamma_minus_2 + 2.0
    gamma_sig = (intrinsic_scatter**2.0 + np.abs(log_R_eff)*delta_slope**2.0 + delta_intercept**2.0)**0.5
    scatter = np.random.randn()*gamma_sig
    return gamma + scatter

def axis_ratio_from_SDSS(vel_disp, A=0.38, B=5.7*1.e-4, truncate=0.2):
    """Sample (one minus) the axis ratio of the lens galaxy from the Rayleigh distribution with scale
    that depends on velocity dispersion

    Parameters
    ----------
    vel_disp : float
        velocity dispersion in km/s

    Note
    ----
    The shape of the distribution arises because more massive galaxies are closer to spherical than 
    less massive ones. The truncation excludes highly-flattened profiles. 
    The default fit values have been derived by [1]_ from the SDSS data. 

    References
    ----------
    .. [1] Collett, Thomas E. 
    "The population of galaxy–galaxy strong lenses in forthcoming optical imaging surveys." 
    The Astrophysical Journal 811.1 (2015): 20.

    Returns
    -------
    float
        the axis ratio q

    """
    scale = A + B*vel_disp
    q = 0.0
    while q < truncate:
        q = 1.0 - np.random.rayleigh(scale, size=None)
    return q

def redshift_binned_luminosity_function(redshift, M_grid):
    """Sample FUV absolute magnitude from the redshift-binned luminosity function

    Parameters
    ----------
    redshift : float

    Note
    ----
    For z < 4, we use the Schechter function fits in Table 1 of [1]_ and,
    for 4 < z < 8, those in Table 4 of [2]_.
    z > 8 are binned into the z=8 bin. I might add high-redshift models, e.g. from [3]_.

    References
    ----------
    .. [1] Arnouts, Stephane, et al. 
    "The GALEX VIMOS-VLT Deep Survey* Measurement of the Evolution of the 1500 Å Luminosity Function." 
    The Astrophysical Journal Letters 619.1 (2005): L43.

    .. [2] Finkelstein, Steven L., et al. 
    "The evolution of the galaxy rest-frame ultraviolet luminosity function over the first two billion years." 
    The Astrophysical Journal 810.1 (2015): 71.

    .. [3] Kawamata, Ryota, et al. 
    "Size–Luminosity Relations and UV Luminosity Functions at z= 6–9 Simultaneously Derived from the Complete Hubble Frontier Fields Data." 
    The Astrophysical Journal 855.1 (2018): 4.

    Returns
    -------
    array-like
        unnormalized function of the absolute magnitude at 1500A

    """
    #prefactor = np.log(10)*phi_star # just normalization
    # Define redshift bins by right edge of bin
    z_bins = np.array([0.2, 0.4, 0.6, 0.8, 1.2, 2.25, 3.4, 4.5, 5.5, 6.5, 7.5, np.inf])
    alphas = np.array([-1.21, -1.19, -1.55, -1.60, -1.63, -1.49, -1.47, -1.56, -1.67, -2.02, -2.03, -2.36])
    M_stars = np.array([-18.05, -18.38, -19.49, -19.84, -20.11, -20.33, -21.08, -20.73, -20.81, -21.13, -21.03, -20.89])
    alpha = alphas[redshift < z_bins][0]
    M_star = M_stars[redshift < z_bins][0]

    # Note phi_star is ignored as normalization
    # Schechter kernel
    exponent = 10.0**(0.4*(M_star - M_grid))
    density = np.exp(-exponent) * exponent**(alpha + 1.0)
    return density

def size_from_luminosity_and_redshift_relation(z, M_V):
    """Sample the effective radius of Lyman break galaxies from the relation with luminosity and redshift

    Parameters
    ----------
    z : float
        redshift
    M_V : float
        V-band absolute magnitude

    Note
    ----
    The relation and scatter agree with [1]_ and [2]_, which both show that size decreases
    with higher redshift. They have been used in LensPop ([3]_).

    References
    ----------
    .. [1] Mosleh, Moein, et al. 
    "The evolution of mass-size relation for Lyman break galaxies from z= 1 to z= 7." 
    The Astrophysical Journal Letters 756.1 (2012): L12.

    .. [2] Huang, Kuang-Han, et al. 
    "The bivariate size-luminosity relations for Lyman break galaxies at z∼ 4-5." 
    The Astrophysical Journal 765.1 (2013): 68.

    .. [3] Collett, Thomas E. 
    "The population of galaxy–galaxy strong lenses in forthcoming optical imaging surveys." 
    The Astrophysical Journal 811.1 (2015): 20.

    Returns
    -------
    float
        a sampled effective radius in kpc

    """
    log_R_eff = (M_V/-19.5)**-0.22 * ((1.0 + z)/5.0)**-1.2
    scatter = np.random.randn()*0.3
    return log_R_eff + scatter

def axis_ratio_disklike(scale=0.3, truncate=0.2):
    """Sample (one minus) the axis ratio of a disk-like galaxy from the Rayleigh distribution

    Note
    ----
    The default parameters are used in Lenspop ([1]_) and are expected for elliptical sources.

    References
    ----------
    .. [1] Collett, Thomas E. 
    "The population of galaxy–galaxy strong lenses in forthcoming optical imaging surveys." 
    The Astrophysical Journal 811.1 (2015): 20.

    Returns
    -------
    float
        the axis ratio

    """
    q = 0.0
    while q < truncate:
        q = 1.0 - np.random.rayleigh(scale, size=None)
    return q
