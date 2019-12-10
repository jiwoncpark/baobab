import numpy as np
from astropy.cosmology import wCDM
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
        raise ValueError("{:s} must be specified in the config inside {:s} for {:s}".format(missing_key, parent_config_key, bnn_prior_class))

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
        sampled_z = np.random.choice(z_grid, 2, replace=False, p=dVol_dz_normed)
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
        z_lens = self.sample_param(redshifts_cfg.z_lens.copy())
        z_src = self.sample_param(redshifts_cfg.z_src.copy())
        while z_src < z_lens + redshifts_cfg.min_diff:
            z_lens = self.sample_param(redshifts_cfg.z_lens.copy())
            z_src = self.sample_param(redshifts_cfg.z_src.copy())
        return z_lens, z_src