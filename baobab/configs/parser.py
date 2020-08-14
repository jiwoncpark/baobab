import os, sys
from datetime import datetime
import warnings
from importlib import import_module
from addict import Dict
import json
from collections import OrderedDict 
import lenstronomy.SimulationAPI.ObservationConfig as obs_cfg

class BaobabConfig:
    """Nested dictionary representing the configuration for Baobab data generation
    
    """
    def __init__(self, user_cfg):
        """
        Parameters
        ----------
        user_cfg : dict or Dict
            user-defined configuration
        
        """
        self.__dict__ = Dict(user_cfg)
        if not hasattr(self, 'out_dir'):
            # Default out_dir path if not specified
            self.out_dir = os.path.join(self.destination_dir, '{:s}_{:s}_prior={:s}_seed={:d}'.format(self.name, self.train_vs_val, self.bnn_prior_class, self.seed))
        self.out_dir = os.path.abspath(self.out_dir)
        if not hasattr(self, 'checkpoint_interval'):
            self.checkpoint_interval = max(100, self.n_data // 100)
        self.get_survey_info(self.survey_info, self.psf.type)
        self.interpret_magnification_cfg()
        self.interpret_kinematics_cfg()
        self.log_filename = datetime.now().strftime("log_%m-%d-%Y_%H:%M_baobab.json")
        self.log_path = os.path.join(self.out_dir, self.log_filename)

    @classmethod
    def from_file(cls, user_cfg_path):
        """Alternative constructor that accepts the path to the user-defined configuration python file
        Parameters
        ----------
        user_cfg_path : str or os.path object
            path to the user-defined configuration python file
        """
        dirname, filename = os.path.split(os.path.abspath(user_cfg_path))
        module_name, ext = os.path.splitext(filename)
        sys.path.insert(0, dirname)
        if ext == '.py':
            #user_cfg_file = map(__import__, module_name)
            #user_cfg = getattr(user_cfg_file, 'cfg')
            user_cfg_script = import_module(module_name)
            user_cfg = getattr(user_cfg_script, 'cfg').deepcopy()
            return cls(user_cfg)
        elif ext == '.json':
            with open(user_cfg_path, 'r') as f:
                user_cfg_str = f.read()
            user_cfg = Dict(json.loads(user_cfg_str)).deepcopy()
            return cls(user_cfg)
        else:
            raise NotImplementedError("This extension is not supported.")

    def export_log(self):
        """Export the baobab log to the current working directory

        """
        with open(self.log_path, 'w') as f:
            json.dump(self.__dict__, f)
            print("Exporting baobab log to {:s}".format(self.log_path))

    def interpret_magnification_cfg(self):
        if 'agn_light' not in self.components:
            if len(self.bnn_omega.magnification.frac_err_sigma) != 0: # non-empty dictionary
                warnings.warn("`bnn_omega.magnification.frac_err_sigma` field is ignored as the images do not contain AGN.")
                self.bnn_omega.magnification.frac_err_sigma = 0.0
        else:
            if 'magnification' not in self.bnn_omega:
                self.bnn_omega.magnification.frac_err_sigma = 0.0
            elif self.bnn_omega.magnification is None:
                self.bnn_omega.magnification.frac_err_sigma = 0.0
            
        if ('magnification' not in self.bnn_omega) and 'agn_light' in self.components:
            self.bnn_omega.magnification.frac_err_sigma = 0.0

    def interpret_kinematics_cfg(self):
        """Validate the kinematics config

        """
        kinematics_cfg = self.bnn_omega.kinematics_cfg
        if kinematics_cfg.anisotropy_model == 'analytic':
            warnings.warn("Since velocity dispersion computation is analytic, any entry other than `sampling_number` in `kinematics.numerics_kwargs` will be ignored.")

    def get_survey_info(self, survey_info, psf_type):
        """Fetch the camera and instrument information corresponding to the survey string identifier

        """
        sys.path.insert(0, obs_cfg.__path__[0])
        survey_module = import_module(survey_info['survey_name'])
        survey_class = getattr(survey_module, survey_info['survey_name'])
        coadd_years = survey_info['coadd_years'] if 'coadd_years' in survey_info else None

        self.survey_object_dict = OrderedDict()
        for bp in survey_info['bandpass_list']:
            survey_object = survey_class(band=bp, psf_type=psf_type, coadd_years=coadd_years)
            # Overwrite ObservationConfig PSF type with user-configured PSF type
            if hasattr(self, 'psf'):
                survey_object.obs['psf_type'] = self.psf.type
            if survey_object.obs['psf_type'] == 'PIXEL':
                if hasattr(self, 'psf'):
                    if hasattr(self.psf, 'psf_kernel_size'):
                        survey_object.psf_kernel_size = self.psf.kernel_size
                    else:
                        raise ValueError("Observation dictionary must specify PSF kernel size if psf_type is PIXEL.")
                    if hasattr(self.psf, 'which_psf_maps'):
                        survey_object.which_psf_maps = self.psf.which_psf_maps
                    else:
                        raise ValueError("Observation dictionary must specify indices of PSF kernel maps if psf_type is PIXEL.")
                else:
                    raise ValueError("User must supply PSF kwargs in the Baobab config if PSF type is PIXEL.")
            else: # 'GAUSSIAN'
                survey_object.psf_kernel_size = None
                survey_object.which_psf_maps = None
            # Override default survey specs with user-specified kwargs
            survey_object.camera.update(survey_info['override_camera_kwargs'])
            survey_object.obs.update(survey_info['override_obs_kwargs'])
            self.survey_object_dict[bp] = survey_object
        # Camera dict is same across bands, so arbitrarily take the last band
        self.instrument = survey_object.camera

    def get_noise_kwargs(self,bandpass):
        """
        Return the noise kwargs defined in the babobab config, e.g. for passing to the noise model for online data augmentation

        Returns
        -------
            (dict): A dict containing the noise kwargs to be passed to the noise
                model.
            (str): The bandpass to pull the noise information for

        """
        # Go through the baobab config and pull out the noise kwargs one by one.
        noise_kwargs = {}
        noise_kwargs.update(self.instrument)
        noise_kwargs.update(self.survey_object_dict[bandpass].kwargs_single_band())
        return noise_kwargs
