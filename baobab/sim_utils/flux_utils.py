import copy
import numpy as np

__all__ = ['mag_to_amp_extended', 'mag_to_amp_point', 'get_unlensed_total_flux', 'get_lensed_total_flux', 'get_unlensed_total_flux_numerical', "calculate_sky_brightness"]

def calculate_sky_brightness(flux_density=7.26*1e-19, lam_eff=15269.1):
    """Calculate the sky brightness in mag with our zeropoint

    Parameters
    ----------
    flux_density: float
        the flux density in cgs units, defined per wavelength in angstroms (Default: 7.26*1e-19, estimated for 12500 ang and taken from Givialisco et al 2002)
    lam_eff : float
        the effective filter wavelength in angstroms (Default: for WFC3/IR F160W, 15279.1 ang). Taken from http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?id=HST/WFC3_IR.F160W

    Note
    ----
    Zeropoint not necessary with absolute flux density values. Only when using cps.

    """
    # All spectral flux density units are per asec^2
    # Zodiacal bg at the North Ecliptic Pole for WFC3, from Giavalisco et al 2002
    flux_density_cgs_wave = flux_density #1.81*1e-18 # erg/cm^2/s^2/ang^1
    # Convert spectral flux density for unit wavelength to that for unit frequency
    flux_density_Jy = flux_density_cgs_wave*(3.34*1e4)*lam_eff**2.0 # Jy
    mag_AB = -2.5*np.log10(flux_density_Jy) + 8.90
    return mag_AB

def mag_to_amp_extended(mag_kwargs_list, light_model, data_api):
    """Convert the magnitude entries into amp (counts per second)
    used by lenstronomy to render the image, for extended objects

    Parameters
    ----------
    mag_kwargs_list : list
        list of kwargs dictionaries in which 'amp' keys are replaced by 'magnitude'
    light_model : lenstronomy.LightModel object
        light model describing the surface brightness profile, used for calculating
        the total flux. Note that only some profiles with an analytic integral can be
        used.
    data_api : lenstronomy.DataAPI object
        a wrapper object around lenstronomy.Observation that has the magnitude zeropoint
        information, with which the magnitude-to-amp conversion is done.

    Returns
    -------
    list
        list of kwargs dictionaries with 'magnitude' replaced by 'amp'

    """
    amp_kwargs_list = copy.deepcopy(mag_kwargs_list)
    for i, mag_kwargs in enumerate(mag_kwargs_list):
        amp_kwargs = amp_kwargs_list[i]
        mag = amp_kwargs.pop('magnitude')
        cps_norm = light_model.total_flux(amp_kwargs_list, norm=True, k=i)[0] # computes the total surface brightness with amp = 1
        cps = data_api.magnitude2cps(mag)
        amp = cps/ cps_norm
        amp_kwargs['amp'] = amp 
    return amp_kwargs_list

def mag_to_amp_point(mag_kwargs_list, point_source_model, data_api):
    """Convert the magnitude entries into amp (counts per second)
    used by lenstronomy to render the image, for point sources

    See the docstring for `mag_to_amp_extended` for parameter descriptions.

    """
    amp_kwargs_list = copy.deepcopy(mag_kwargs_list)
    amp_list = []
    for i, mag_kwargs in enumerate(mag_kwargs_list):
        amp_kwargs = amp_kwargs_list[i]
        mag = np.array(amp_kwargs.pop('magnitude'))
        cps_norm = 1.0
        cps = data_api.magnitude2cps(mag)
        amp = cps/ cps_norm
        amp_list.append(amp)
    amp_kwargs_list = point_source_model.set_amplitudes(amp_list, amp_kwargs_list)
    return amp_kwargs_list

def get_unlensed_total_flux(kwargs_src_light_list, src_light_model, kwargs_ps_list=None):
    """Compute the total flux of unlensed objects

    Parameter
    ---------
    kwargs_src_light_list : list
        list of kwargs dictionaries for the unlensed source galaxy, each with an 'amp' key
    kwargs_ps_list : list
        list of kwargs dictionaries for the unlensed point source (if any), each with an 'amp' key

    Returns
    -------
    float
        the total unlensed flux

    """
    total_flux = 0.0
    for i, kwargs_src in enumerate(kwargs_src_light_list):
        total_flux += src_light_model.total_flux(kwargs_src_light_list, norm=True, k=i)[0]
    if kwargs_ps_list is not None:
        for i, kwargs_ps in enumerate(kwargs_ps_list):
            total_flux += kwargs_ps['point_amp']
    return total_flux

def get_unlensed_total_flux_numerical(kwargs_src_light, kwargs_ps, image_model, return_image=False):
    """Compute the total flux of the unlensed image by rendering the source on a pixel grid

    Returns
    -------
    float
        the total unlensed flux

    """
    unlensed_src_image = image_model.image(kwargs_lens=None, kwargs_source=kwargs_src_light, kwargs_lens_light=None, kwargs_ps=kwargs_ps, lens_light_add=False)
    unlensed_total_flux = np.sum(unlensed_src_image)
    if return_image:
        return unlensed_total_flux, unlensed_src_image
    else:
        return unlensed_total_flux
        
def get_lensed_total_flux(kwargs_lens_mass, kwargs_src_light, kwargs_ps, image_model, return_image=False):
    """Compute the total flux of the lensed image

    Returns
    -------
    float
        the total lensed flux

    """

    lensed_src_image = image_model.image(kwargs_lens_mass, kwargs_source=kwargs_src_light, kwargs_lens_light=None, kwargs_ps=kwargs_ps, lens_light_add=False, point_source_add=True if kwargs_ps is not None else False)
    lensed_total_flux = np.sum(lensed_src_image)
    if return_image:
        return lensed_total_flux, lensed_src_image
    else:
        return lensed_total_flux