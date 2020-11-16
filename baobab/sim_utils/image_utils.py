import copy
import numpy as np
# Lenstronomy modules
from lenstronomy.ImSim.image_model import ImageModel
from baobab.sim_utils import mag_to_amp_extended, mag_to_amp_point, get_lensed_total_flux, get_unlensed_total_flux_numerical
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.SimulationAPI.data_api import DataAPI
from lenstronomy.PointSource.point_source import PointSource
from baobab.sim_utils import psf_utils


__all__ = ['Imager']

class Imager:
    """Deterministic utility class for imaging the objects on a pixel grid

        Attributes
        ----------
        bnn_omega : dict
            copy of `cfg.bnn_omega`
        components : list
            list of components, e.g. `lens_mass`

        """
    def __init__(self, components, lens_mass_model, src_light_model, lens_light_model=None, ps_model=None, kwargs_numerics={'supersampling_factor': 1}, min_magnification=0.0, for_cosmography=False, magnification_frac_err=0.0):
        self.components = components
        self.kwargs_numerics = kwargs_numerics
        self.lens_mass_model = lens_mass_model
        self.src_light_model = src_light_model
        self.lens_light_model = lens_light_model
        self.ps_model = ps_model
        self.unlensed_ps_model = PointSource(point_source_type_list=['SOURCE_POSITION'], fixed_magnification_list=[False])
        self.lens_eq_solver = LensEquationSolver(self.lens_mass_model)
        self.min_magnification = min_magnification
        self.for_cosmography = for_cosmography
        self.magnification_frac_err = magnification_frac_err
        self.img_features = {} # Initialized to store metadata of images, will get updated for each lens

    def _set_sim_api(self, num_pix, kwargs_detector, psf_kernel_size, which_psf_maps):
        """Set the simulation API objects

        """
        self.data_api = DataAPI(num_pix, **kwargs_detector)
        #self.pixel_scale = data_api.pixel_scale
        pixel_scale = kwargs_detector['pixel_scale']
        psf_model = psf_utils.get_PSF_model(kwargs_detector['psf_type'], pixel_scale, seeing=kwargs_detector['seeing'], kernel_size=psf_kernel_size, which_psf_maps=which_psf_maps)
        # Set the precision level of lens equation solver
        self.min_distance = 0.05
        self.search_window = pixel_scale*num_pix
        self.image_model = ImageModel(self.data_api.data_class, psf_model, self.lens_mass_model, self.src_light_model, self.lens_light_model, self.ps_model, kwargs_numerics=self.kwargs_numerics)
        if 'agn_light' in self.components:
            self.unlensed_image_model = ImageModel(self.data_api.data_class, psf_model, None, self.src_light_model, None, self.unlensed_ps_model, kwargs_numerics=self.kwargs_numerics)
        else:
            self.unlensed_image_model = ImageModel(self.data_api.data_class, psf_model, None, self.src_light_model, None, None, kwargs_numerics=self.kwargs_numerics)

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
            self.kwargs_unlensed_unmagnified_amp_ps = None

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
                                                                   num_iter_max=100, # default is 10 but td_cosmography default is 100
                                                                   precision_limit=10**(-10) # default for both this and td_cosmography
                                                                    ) 
        agn_light_sample = sample['agn_light']
        unlensed_mag = agn_light_sample['magnitude'] # unlensed agn mag
        # Save the unlensed (source-plane) kwargs in amplitude units
        kwargs_unlensed_unmagnified_mag_ps = [{'ra_source': self.kwargs_src_light[0]['center_x'], 'dec_source': self.kwargs_src_light[0]['center_y'], 'magnitude': unlensed_mag}]
        self.kwargs_unlensed_unmagnified_amp_ps = mag_to_amp_point(kwargs_unlensed_unmagnified_mag_ps, self.unlensed_ps_model, self.data_api) # note 
        # Compute the lensed (image-plane), magnified kwargs in amplitude units
        magnification = self.lens_mass_model.magnification(x_image, y_image, kwargs=self.kwargs_lens_mass)
        measured_magnification = np.abs(magnification*(1.0 + self.magnification_frac_err*np.random.randn(len(magnification)))) # Add noise to magnification
        magnification = np.abs(magnification)
        kwargs_lensed_unmagnified_mag_ps = [{'ra_image': x_image, 'dec_image': y_image, 'magnitude': unlensed_mag}] # note unlensed magnitude
        kwargs_lensed_unmagnified_amp_ps = mag_to_amp_point(kwargs_lensed_unmagnified_mag_ps, self.ps_model, self.data_api) # note unmagnified amp
        self.kwargs_ps = copy.deepcopy(kwargs_lensed_unmagnified_amp_ps)
        for kw in self.kwargs_ps:
            kw.update(point_amp=kw['point_amp']*measured_magnification)
        # Log the solved image positions
        self.img_features.update(x_image=x_image, 
                                 y_image=y_image, 
                                 magnification=magnification,
                                 measured_magnification=measured_magnification)

    def generate_image(self, sample, num_pix, survey_object_dict):
        img_canvas = np.empty([len(survey_object_dict), num_pix, num_pix]) # [n_filters, num_pix, num_pix]
        # Loop over bands
        for i, (bp, survey_object) in enumerate(survey_object_dict.items()):
            self._set_sim_api(num_pix, survey_object.kwargs_single_band(), survey_object.psf_kernel_size, survey_object.which_psf_maps)
            self._load_kwargs(sample)
            # Reject nonsensical number of images (due to insufficient numerical precision)
            if ('y_image' in self.img_features) and (len(self.img_features['y_image']) not in [2, 4]):
                return None, None
            # Compute magnification
            lensed_total_flux = get_lensed_total_flux(self.kwargs_lens_mass, self.kwargs_src_light, self.kwargs_ps, self.image_model)
            #unlensed_total_flux = get_unlensed_total_flux(self.kwargs_src_light, self.src_light_model, self.kwargs_unlensed_amp_ps, self.ps_model)
            unlensed_total_flux = get_unlensed_total_flux_numerical(self.kwargs_src_light, self.kwargs_unlensed_unmagnified_amp_ps, self.unlensed_image_model)
            total_magnification = lensed_total_flux/unlensed_total_flux
            # Apply magnification cut
            if (total_magnification < self.min_magnification) or np.isnan(total_magnification):
                return None, None
            # Generate image for export
            img = self.image_model.image(self.kwargs_lens_mass, self.kwargs_src_light, self.kwargs_lens_light, self.kwargs_ps)
            img = np.maximum(0.0, img) # safeguard against negative pixel values
            img_canvas[i, :, :] = img 
            # Save remaining image features
            img_features_single_band = {f'total_magnification_{bp}': total_magnification, f'lensed_total_flux_{bp}': lensed_total_flux, f'unlensed_total_flux_{bp}': unlensed_total_flux}
            self.img_features.update(img_features_single_band)
        return img_canvas, self.img_features

    def add_noise(self, image_array):
        """Add noise to the image (deprecated; replaced by the data_augmentation package)

        """
        #noise_map = self.data_api.noise_for_model(image_array, background_noise=True, poisson_noise=True, seed=None)
        #image_array += noise_map
        #return image_array
        pass