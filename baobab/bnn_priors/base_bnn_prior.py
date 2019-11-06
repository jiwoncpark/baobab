from abc import ABC, abstractmethod
import baobab.distributions

class BaseBNNPrior(ABC):
    """Abstract base class equipped with PDF evaluation and sampling utility functions for various lens/source macromodels

    """
    def __init__(self):
        self.set_required_parameters()

    def set_required_parameters(self):
        """Defines a dictionary of the list of parameters (value) corresponding to each profile (key).

        The parameter names follow the lenstronomy convention.
        The dictionary will be updated as more profiles are supported.

        """
        params = dict(SPEMD=['center_x', 'center_y', 'gamma', 'theta_E', 'e1', 'e2'],
                          SHEAR_GAMMA_PSI=['gamma_ext', 'psi_ext'],
                          SERSIC_ELLIPSE=['magnitude', 'center_x', 'center_y', 'n_sersic', 'R_sersic', 'e1', 'e2'],
                          LENSED_POSITION=['magnitude'],
                          SOURCE_POSITION=['ra_source', 'dec_source', 'magnitude'],)
        setattr(self, 'params', params)

    def _raise_config_error(self, missing_key, parent_config_key, bnn_prior_class):
        """Convenience function for raising errors related to config values

        """
        raise ValueError("{:s} must be specified in the config inside {:s} for {:s}".format(missing_key,
                                                                                             parent_config_key,
                                                                                             bnn_prior_class))
    
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

    @abstractmethod
    def sample(self):
        """Gets kwargs of sampled parameters to be passed to lenstronomy

        Overridden by subclasses.

        """
        return NotImplemented