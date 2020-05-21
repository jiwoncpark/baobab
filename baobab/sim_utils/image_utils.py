import copy
import numpy as np
# Lenstronomy modules
from lenstronomy.ImSim.image_model import ImageModel
from baobab.sim_utils import amp_to_mag_extended, amp_to_mag_point, get_lensed_total_flux, get_unlensed_total_flux
__all__ = ['generate_image', 'generate_image_simple']

def generate_image(sample, psf_model, data_api, lens_mass_model, src_light_model, lens_eq_solver, pixel_scale, num_pix, components, kwargs_numerics, min_magnification=0.0, lens_light_model=None, ps_model=None, reject_unmatching_td=True):
    """Generate an image from provided model and model parameters

    Parameters
    ----------
    sample : dict
        sampled model parameters
    psf_models : lenstronomy PSF object
        the PSF kernel point source map
    data_api : lenstronomy DataAPI object
        tool that handles detector and observation conditions 
    

    Returns
    -------
    tuple of (np.array, dict)
        the image and its features

    """
    img_features = dict()
    image_data = data_api.data_class
    kwargs_lens_mass = [sample['lens_mass'], sample['external_shear']]
    kwargs_src_light = [sample['src_light']]
    kwargs_src_light = amp_to_mag_extended(kwargs_src_light, src_light_model, data_api)
    img_features['src_light_amp'] = kwargs_src_light[0]['amp']
    kwargs_lens_light = None
    kwargs_ps = None
    # Add AGN point source metadata
    if 'agn_light' in components:
        x_image, y_image = lens_eq_solver.findBrightImage(sample['src_light']['center_x'],
                                                          sample['src_light']['center_y'],
                                                          kwargs_lens_mass,
                                                          min_distance=0.01, # default is 0.01 but td_cosmography default is 0.05
                                                          numImages=4,
                                                          search_window=5, #num_pix*pixel_scale, # default is 5 for both this and td_cosmography
                                                          #num_iter_max=10, # default is 10 but td_cosmography default is 100
                                                          precision_limit=10**(-10) # default for both this and td_cosmography
                                                          ) 
        magnification = np.abs(lens_mass_model.magnification(x_image, y_image, kwargs=kwargs_lens_mass))
        unlensed_mag = sample['agn_light']['magnitude'] # unlensed agn mag
        kwargs_unlensed_mag_ps = [{'ra_image': x_image, 'dec_image': y_image, 'magnitude': unlensed_mag}] # note unlensed magnitude
        kwargs_unlensed_amp_ps = amp_to_mag_point(kwargs_unlensed_mag_ps, ps_model, data_api) # note unlensed amp
        kwargs_ps = copy.deepcopy(kwargs_unlensed_amp_ps)
        for kw in kwargs_ps:
            kw.update(point_amp=kw['point_amp']*magnification)
        img_features['x_image'] = x_image
        img_features['y_image'] = y_image
        #if 'true_td' in sample['misc'] and reject_unmatching_td:
        #    if len(x_image) != len(np.trim_zeros(sample['misc']['true_td'], 'b')):
        #        print("match td length cut")
        #        # Depending on the numerics of the time delay calculation and image finder, the number of images may not agree. Reject these examples.
        #        return None, None
    else:
        kwargs_unlensed_amp_ps = None
    # Add lens light metadata
    if 'lens_light' in components:
        kwargs_lens_light = [sample['lens_light']]
        kwargs_lens_light = amp_to_mag_extended(kwargs_lens_light, lens_light_model, data_api)
    # Instantiate image model
    image_model = ImageModel(image_data, psf_model, lens_mass_model, src_light_model, lens_light_model, ps_model, kwargs_numerics=kwargs_numerics)
    # Compute magnification
    lensed_total_flux = get_lensed_total_flux(kwargs_lens_mass, kwargs_src_light, kwargs_lens_light, kwargs_ps, image_model)
    unlensed_total_flux = get_unlensed_total_flux(kwargs_src_light, src_light_model, kwargs_unlensed_amp_ps, ps_model)
    total_magnification = lensed_total_flux/unlensed_total_flux
    # Apply magnification cut
    if (total_magnification < min_magnification) or np.isnan(total_magnification):
        return None, None
    # Generate image for export
    img = image_model.image(kwargs_lens_mass, kwargs_src_light, kwargs_lens_light, kwargs_ps)
    # Add noise
    #if add_noise:
    #    noise = data_api.noise_for_model(img, background_noise=True, poisson_noise=True, seed=None)
    #    img += noise
    img = np.maximum(0.0, img) # safeguard against negative pixel values
    # Save remaining image features
    img_features['total_magnification'] = total_magnification

    return img, img_features

def generate_image_simple(sample, psf_model, data_api, lens_mass_model, src_light_model, lens_eq_solver, pixel_scale, num_pix, components, kwargs_numerics, min_magnification=0.0, lens_light_model=None, ps_model=None):
    """Generate an image from provided model and model parameters

    Parameters
    ----------
    sample : dict
        sampled model parameters
    psf_models : lenstronomy PSF object
        the PSF kernel point source map
    data_api : lenstronomy DataAPI object
        tool that handles detector and observation conditions 
    
    Note
    ----
    kwargs must have 'amp' not 'magnitude'.

    Returns
    -------
    tuple of (np.array, dict)
        the image and its features

    """
    img_features = dict()
    image_data = data_api.data_class
    kwargs_lens_mass = [sample['lens_mass'], sample['external_shear']]
    kwargs_src_light = [sample['src_light']]
    kwargs_lens_light = None
    kwargs_ps = None
    # Add AGN point source metadata
    if 'agn_light' in components:
        x_image, y_image = lens_eq_solver.findBrightImage(sample['src_light']['center_x'],
                                                          sample['src_light']['center_y'],
                                                          kwargs_lens_mass,
                                                          min_distance=0.01, # default is 0.01 but td_cosmography default is 0.05
                                                          numImages=4,
                                                          search_window=num_pix*pixel_scale, #num_pix*pixel_scale, # default is 5 for both this and td_cosmography
                                                          num_iter_max=100, # default is 10 but td_cosmography default is 100
                                                          precision_limit=10**(-10) # default for both this and td_cosmography
                                                          )
        magnification = np.abs(lens_mass_model.magnification(x_image, y_image, kwargs=kwargs_lens_mass))
        unlensed_mag = sample['agn_light']['magnitude'] # unlensed agn mag
        kwargs_unlensed_mag_ps = [{'ra_image': x_image, 'dec_image': y_image, 'magnitude': unlensed_mag}] # note unlensed magnitude
        kwargs_unlensed_amp_ps = amp_to_mag_point(kwargs_unlensed_mag_ps, ps_model, data_api) # note unlensed amp
        kwargs_ps = copy.deepcopy(kwargs_unlensed_amp_ps)
        for kw in kwargs_ps:
            kw.update(point_amp=kw['point_amp']*magnification)
        img_features['x_image'] = x_image
        img_features['y_image'] = y_image
    else:
        kwargs_unlensed_amp_ps = None
    # Add lens light metadata
    if 'lens_light' in components:
        kwargs_lens_light = [sample['lens_light']]
    # Instantiate image model
    image_model = ImageModel(image_data, psf_model, lens_mass_model, src_light_model, lens_light_model, ps_model, kwargs_numerics=kwargs_numerics)
    # Generate image for export
    img = image_model.image(kwargs_lens_mass, kwargs_src_light, kwargs_lens_light, kwargs_ps)
    # Add noise
    #if add_noise:
    #    noise = data_api.noise_for_model(img, background_noise=True, poisson_noise=True, seed=None)
    #    img += noise
    img = np.maximum(0.0, img) # safeguard against negative pixel values
    # Save remaining image features

    return img, img_features