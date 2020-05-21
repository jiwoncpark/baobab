__all__ = ['velocity_dispersion_analytic', 'velocity_dispersion_numerical']

def velocity_dispersion_analytic(td_cosmo_object, kwargs_lens, kwargs_lens_light, kwargs_anisotropy, kwargs_aperture, kwargs_psf, anisotropy_model, r_eff, kwargs_numerics, kappa_ext):
    """Get the LOS velocity dispersion of the lens within a square slit of given width and length and seeing with the given FWHM. The computation is analytic as it assumes a Hernquist light profiel and a spherical power-law lens model at the first position.

    Parameters
    ----------
    td_cosmo_object : `lenstronomy.Analysis.TDCosmography` object
        tool with which to compute the velocity dispersion
    kwargs_lens : list of dict
        lens mass parameters
    kwargs_lens_light : list of dict
        lens light parameters
    kwargs_anisotropy : dict
        anisotropy parameters such as `r_ani`
    kwargs_aperture : dict
        aperture geometry
    kwargs_psf : dict
        seeing conditions
    anisotropy_model : str
        `analytic` if using this module, else the model to evaluate numerically, e.g. `OsipkovMerritt`
    r_eff : float
        rough estimate of the half-light radius of the lens light
    kwargs_numerics : dict
        numerical solver config

    Returns
    -------
    float
        the sigma of the velocity dispersion
        
    """
    module = getattr(td_cosmo_object, 'velocity_dispersion_analytical')
    vel_disp = module(
                      theta_E=kwargs_lens[0]['theta_E'],
                      gamma=kwargs_lens[0]['gamma'],
                      r_ani=kwargs_anisotropy['aniso_param']*r_eff,
                      r_eff=r_eff,
                      kwargs_aperture=kwargs_aperture,
                      kwargs_psf=kwargs_psf,
                      sampling_number=kwargs_numerics['sampling_number'],
                      kappa_ext=kappa_ext,
                      )
    return vel_disp

def velocity_dispersion_numerical(td_cosmo_object, kwargs_lens, kwargs_lens_light, kwargs_anisotropy, kwargs_aperture, kwargs_psf, anisotropy_model, r_eff, kwargs_numerics, kappa_ext):
    """Get the velocity dispersion using a numerical model

    See `velocity_dispersion_analytic` for the parameter description.

    """
    module = getattr(td_cosmo_object, 'velocity_dispersion_numerical')
    vel_disp = module(
                      kwargs_lens=kwargs_lens,
                      kwargs_lens_light=kwargs_lens_light,
                      kwargs_anisotropy={'r_ani': kwargs_anisotropy['aniso_param']*r_eff},
                      kwargs_aperture=kwargs_aperture,
                      kwargs_psf=kwargs_psf,
                      MGE_light=False,
                      kwargs_mge_light=False,
                      MGE_mass=False,
                      kwargs_mge_mass=False,
                      Hernquist_approx=False,
                      anisotropy_model=anisotropy_model,
                      r_eff=r_eff,
                      kwargs_numerics=kwargs_numerics,
                      kappa_ext=kappa_ext,
                      )
    return vel_disp