import numpy as np
from astropy.cosmology import wCDM
import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.Analysis.lens_properties import LensProp
from abc import ABC, abstractmethod

class BaseCosmoBNNPrior(ABC):
    """Abstract base class for a cosmology-aware BNN prior

    """
    def __init__(self, bnn_omega):
        self._check_cosmology_config_validity(bnn_omega)
        self._define_cosmology(bnn_omega.cosmology)
        for cosmo_comp in ['cosmology', 'redshift', 'kinematics']:
            setattr(self, cosmo_comp, bnn_omega[cosmo_comp])

        self.sample_redshifts = getattr(self, 'sample_redshifts_from_{:s}'.format(self.redshift.model))

    def _raise_config_error(self, missing_key, parent_config_key, bnn_prior_class):
        """Convenience function for raising errors related to config values

        """
        raise ValueError("{:s} must be specified in the config inside {:s} for {:s}".format(missing_key,
                                                                                             parent_config_key,
                                                                                             bnn_prior_class))

    def _check_cosmology_config_validity(self, bnn_omega):
        """Check whether the config file specified the hyperparameters for all the fields
        required for cosmology-aware BNN priors, e.g. cosmology, redshift, galaxy kinematics

        """
        required_keys = ['cosmology', 'redshift', 'kinematics']
        for possible_missing_key in required_keys:
            if possible_missing_key not in bnn_omega:
                self._raise_cfg_error(possible_missing_key, 'bnn_omega', cls.__name__)

    def _define_cosmology(self, cosmology_cfg):
        """Set the cosmology, with which to generate all the training samples, based on the config

        Parameters
        ----------
        cosmology_cfg : dict
            Copy of `cfg.bnn_omega.cosmology`

        """
        self.cosmo = wCDM(**cosmology_cfg)

    def sample_param(self, hyperparams):
        """Assigns a sampling distribution

        """
        dist = hyperparams.pop('dist')
        return getattr(baobab.distributions, 'sample_{:s}'.format(dist))(**hyperparams)

    def eval_param_pdf(self, eval_at, hyperparams):
        """Assigns and evaluates the PDF 

        """
        dist = hyperparams.pop('dist')
        return getattr(baobab.distributions, 'eval_{:s}_pdf'.format(dist))(**hyperparams)

    def sample_redshifts_from_differential_comoving_volume(self, redshifts_cfg):
        """Sample redshifts from the differential comoving volume,
        on a grid with the range and resolution specified in the config

        Parameters
        ----------
        redshifts_cfg : dict
            Copy of `cfg.bnn_omega.redshift`

        Returns
        -------
        tuple
            the tuple of floats that are the realized z_lens, z_src

        """
        z_grid = np.arange(**redshifts_cfg.grid)
        dVol_dz = self.cosmo.differential_comoving_volume(z_grid).value
        dVol_dz_normed = dVol_dz/np.sum(dVol_dz)
        sampled_z = np.random.choice(z_grid, 2, replace=True, p=dVol_dz_normed)
        z_lens = np.min(sampled_z)
        z_src = np.max(sampled_z)
        return z_lens, z_src

    def sample_redshifts_from_independent_dist(self, redshifts_cfg):
        """Sample lens and source redshifts from independent distributions, while enforcing that the lens redshift is smaller than source redshift

        Parameters
        ----------
        redshifts_cfg : dict
            Copy of `cfg.bnn_omega.redshift`

        Returns
        -------
        tuple
            the tuple of floats that are the realized z_lens, z_src

        """
        z_lens = self.sample_param(**redshifts_cfg.z_lens)
        z_src = self.sample_param(**redshifts_cfg.z_src)
        while z_lens > z_src:
            z_lens = self.sample_param(**redshifts_cfg.z_lens)
            z_src = self.sample_param(**redshifts_cfg.z_src)
        return z_lens, z_src

    def instantiate_lens_prop(self, kwargs_model_list, z_lens, z_src):
        """Instantiate the LensProp class, used for getting time delays and velocity dispersions
        
        Parameters
        ----------
        kwargs_model_list : dict
            type of lenstronomy-supported paramerizations for each component 
        z_lens : float
        z_src : float

        Returns
        -------
        lenstronomy LensProp object

        """
        lens_prop = LensProp(z_lens, z_src, kwargs_model_list, self.cosmo)
        return lens_prop

    def get_true_vel_disp(self, lens_prop, kwargs_lens, r_eff, kappa_ext, kinematics_cfg):
        """Infer the velocity dispersion given the lens model, using the spherical Jeans model
        
        Parameters
        ----------
        lens_prop : lenstronomy LensProp object
        kwargs_lens : list of dict
            the parameter values of the profiles listed in `kwargs_model_list['lens_model_list']`
        r_eff : float
            lens effective radius in arcsec
        kinematics_cfg : dict
            copy of `cfg.bnn_omega.kinematics`

        Returns
        -------
        float
            the inferred velocity dispersion in km/s

        """
        inferred_vd = lens_prop.velocity_dispersion(kwargs_lens,
                                                    r_eff=r_eff,
                                                    R_slit=kinematics_cfg.horizontal_aperture_size,
                                                    dR_slit=kinematics_cfg.vertical_aperture_size,
                                                    psf_fwhm=kinematics_cfg.psf_fwhm_eff,
                                                    aniso_param=kinematics_cfg.aniso_param,
                                                    num_evaluate=kinematics_cfg.num_evaluate,
                                                    kappa_ext=kappa_ext
                                                    )

    def get_measured_vel_disp(self, true_vel_disp, vel_disp_err):
        """Get the velocity dispersion with measurement error

        Parameters
        ----------
        true_vel_disp : float
            true value of the velocity dispersion in km/s
        vel_disp_err : float
            the measurement error on the velocity dispersion to apply, in km/s

        Returns
        -------
        float
            the measured velocity dispersion in km/s

        """
        return true_vel_disp + np.random.randn()*vel_disp_err

    def get_measured_time_delays(self, ):
        pass
