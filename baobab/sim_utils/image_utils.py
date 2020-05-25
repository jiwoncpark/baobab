import copy
import numpy as np
# Lenstronomy modules
from lenstronomy.ImSim.image_model import ImageModel
from baobab.sim_utils import mag_to_amp_extended, mag_to_amp_point, get_lensed_total_flux, get_unlensed_total_flux
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.SimulationAPI.data_api import DataAPI


__all__ = ['Imager', 'generate_image', 'generate_image_simple']

class Imager:
    """Deterministic utility class for imaging the objects on a pixel grid

        Attributes
        ----------
        bnn_omega : dict
            copy of `cfg.bnn_omega`
        components : list
            list of components, e.g. `lens_mass`

        """
    def __init__(self, components, lens_mass_model, src_light_model, lens_light_model=None, ps_model=None, kwargs_numerics={'supersampling_factor': 1}, min_magnification=0.0, for_cosmography=False):
        self.components = components
        self.kwargs_numerics = kwargs_numerics
        self.lens_mass_model = lens_mass_model
        self.src_light_model = src_light_model
        self.lens_light_model = lens_light_model
        self.ps_model = ps_model
        self.min_distance = 0.01
        self.search_window = 5.0
        self.lens_eq_solver = LensEquationSolver(self.lens_mass_model)
        self.min_magnification = min_magnification
        self.for_cosmography = for_cosmography

    def _set_sim_api(self, num_pix, kwargs_detector):
        """Set the simulation API objects

        """
        self.data_api = DataAPI(num_pix, **kwargs_detector)
        #self.pixel_scale = data_api.pixel_scale
        psf_model = kwargs_detector['kernel_point_source']
        self.image_model = ImageModel(self.data_api.data_class, psf_model, self.lens_mass_model, self.src_light_model, self.lens_light_model, self.ps_model, kwargs_numerics=self.kwargs_numerics)

    def _load_kwargs(self, sample):
        """Generate an image from provided model and model parameters

        Parameters
        ----------
        sample : dict
            model parameters sampled by a bnn_prior object

        """
        self._load_lens_mass_kwargs(sample['lens_mass'], sample['external_shear'])
        self._load_src_light_kwargs(sample['src_light'])
        if 'lens_light' in self.components:
            self._load_lens_light_kwargs(sample['lens_light'])
        else:
            self.kwargs_lens_light = None
        if 'agn_light' in self.components:
            self._load_agn_light_kwargs(sample)
        else:
            self.kwargs_ps = None
            self.kwargs_unlensed_amp_ps = None

    def _load_lens_mass_kwargs(self, lens_mass_sample, external_shear_sample):
        self.kwargs_lens_mass = [lens_mass_sample, external_shear_sample]

    def _load_src_light_kwargs(self, src_light_sample):
        kwargs_src_light = [src_light_sample]
        # Convert from mag to amp
        self.kwargs_src_light = mag_to_amp_extended(kwargs_src_light, self.src_light_model, self.data_api)

    def _load_lens_light_kwargs(self, lens_light_sample):
        kwargs_lens_light = [lens_light_sample]
        # Convert lens magnitude into amp
        self.kwargs_lens_light = mag_to_amp_extended(kwargs_lens_light, self.lens_light_model, self.data_api)

    def _load_agn_light_kwargs(self, sample):
        """Set the point source kwargs to be ingested by Lenstronomy

        """
        # When using the image positions for cosmological parameter recovery, the time delays must be computed by evaluating the Fermat potential at these exact positions.
        if self.for_cosmography:
            x_image = sample['misc']['x_image']
            y_image = sample['misc']['y_image']
        # When the precision of the lens equation solver doesn't have to be matched between image positions and time delays, simply solve for the image positions using whatever desired precision.
        else:
            x_image, y_image = self.lens_eq_solver.findBrightImage(self.kwargs_src_light[0]['center_x'], 
                                                                   self.kwargs_src_light[0]['center_y'],
                                                                   self.kwargs_lens_mass,
                                                                   min_distance=self.min_distance,
                                                                   search_window=self.search_window,
                                                                   numImages=4,
                                                                   num_iter_max=10, # default is 10 but td_cosmography default is 100
                                                                   precision_limit=10**(-10) # default for both this and td_cosmography
                                                                    ) 
        agn_light_sample = sample['agn_light']
        magnification = np.abs(self.lens_mass_model.magnification(x_image, y_image, kwargs=self.kwargs_lens_mass))
        unlensed_mag = agn_light_sample['magnitude'] # unlensed agn mag
        kwargs_unlensed_mag_ps = [{'ra_image': x_image, 'dec_image': y_image, 'magnitude': unlensed_mag}] # note unlensed magnitude
        self.kwargs_unlensed_amp_ps = mag_to_amp_point(kwargs_unlensed_mag_ps, self.ps_model, self.data_api) # note unlensed amp
        self.kwargs_ps = copy.deepcopy(self.kwargs_unlensed_amp_ps)
        for kw in self.kwargs_ps:
            kw.update(point_amp=kw['point_amp']*magnification)
        # Log the solved image positions
        self.img_features.update(x_image=x_image, y_image=y_image, magnification=magnification)

    def generate_image(self, sample, num_pix, kwargs_detector):
        self._set_sim_api(num_pix, kwargs_detector)
        self.img_features = {} # any metadata computed while generating the images
        self._load_kwargs(sample)
        # Reject nonsensical number of images (due to insufficient numerical precision)
        if len(self.img_features['y_image']) not in [2, 4]:
            return None, None
        # Compute magnification
        lensed_total_flux = get_lensed_total_flux(self.kwargs_lens_mass, self.kwargs_src_light, self.kwargs_lens_light, self.kwargs_ps, self.image_model)
        unlensed_total_flux = get_unlensed_total_flux(self.kwargs_src_light, self.src_light_model, self.kwargs_unlensed_amp_ps, self.ps_model)
        total_magnification = lensed_total_flux/unlensed_total_flux
        # Apply magnification cut
        if (total_magnification < self.min_magnification) or np.isnan(total_magnification):
            return None, None
        # Generate image for export
        img = self.image_model.image(self.kwargs_lens_mass, self.kwargs_src_light, self.kwargs_lens_light, self.kwargs_ps)
        img = np.maximum(0.0, img) # safeguard against negative pixel values
        # Save remaining image features
        self.img_features.update(total_magnification=total_magnification)
        return img, self.img_features

    def add_noise(self, image_array):
        """Add noise to the image (deprecated; replaced by the data_augmentation package)

        """
        #noise_map = self.data_api.noise_for_model(image_array, background_noise=True, poisson_noise=True, seed=None)
        #image_array += noise_map
        #return image_array
        pass

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
    kwargs_src_light = mag_to_amp_extended(kwargs_src_light, src_light_model, data_api)
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
        kwargs_unlensed_amp_ps = mag_to_amp_point(kwargs_unlensed_mag_ps, ps_model, data_api) # note unlensed amp
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
        # Convert lens magnitude into amp
        kwargs_lens_light = mag_to_amp_extended(kwargs_lens_light, lens_light_model, data_api)
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
        kwargs_unlensed_amp_ps = mag_to_amp_point(kwargs_unlensed_mag_ps, ps_model, data_api) # note unlensed amp
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