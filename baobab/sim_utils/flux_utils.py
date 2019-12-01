import copy
import numpy as np
__all__ = ['amp_to_mag_extended', 'amp_to_mag_point', 'get_unlensed_total_flux', 'get_lensed_total_flux']

def amp_to_mag_extended(mag_kwargs_list, light_model, data_api):
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
        cps_norm = light_model.total_flux(amp_kwargs_list, norm=True, k=i)[0]
        cps = data_api.magnitude2cps(mag)
        amp = cps/ cps_norm
        amp_kwargs['amp'] = amp 
    return amp_kwargs_list

def amp_to_mag_point(mag_kwargs_list, point_source_model, data_api):
    """Convert the magnitude entries into amp (counts per second)
    used by lenstronomy to render the image, for point sources

    See the docstring for `amp_to_mag_extended` for parameter descriptions.

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

def get_unlensed_total_flux(kwargs_src_light_list, src_light_model, kwargs_ps_list=None, ps_model=None):
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
        assert ps_model is not None
        for i, kwargs_ps in enumerate(kwargs_ps_list):
            total_flux += kwargs_ps['point_amp']
    return total_flux
        
def get_lensed_total_flux(kwargs_lens_mass, kwargs_src_light, kwargs_lens_light, kwargs_ps, image_model):
    """Compute the total flux of the lensed image

    Returns
    -------
    float
        the total lensed flux

    """

    lensed_src_image = image_model.image(kwargs_lens_mass, kwargs_src_light, kwargs_lens_light, kwargs_ps, lens_light_add=False)
    lensed_total_flux = np.sum(lensed_src_image)
    return lensed_total_flux