import numpy as np
from astropy.cosmology import wCDM
import lenstronomy.Util.param_util as param_util
from abc import ABC, abstractmethod

class BaseCosmoBNNPrior(ABC):
    """Abstract base class for a cosmology-aware BNN prior

    """
    def __init__(self, bnn_omega):
        self._check_cosmology_config_validity(bnn_omega)
        self._define_cosmology(bnn_omega.cosmology)

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

    def sample_redshifts(self, redshifts_cfg):
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


