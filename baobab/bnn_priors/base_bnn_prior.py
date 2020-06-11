from abc import ABC, abstractmethod
import baobab.distributions

class BaseBNNPrior(ABC):
    """Abstract base class equipped with PDF evaluation and sampling utility functions for various lens/source macromodels

    """
    def __init__(self, bnn_omega, components):
        self.components = components
        for comp in bnn_omega:
            setattr(self, comp, bnn_omega[comp])
        self._set_required_parameters()
        self._define_kwargs_model()

    def set_params_list(self, params_to_exclude):
        """Set the list of tuples, each tuple specifying the component and parameter name, to be realized independently as well as the list of tuples to be converted from the q, phi convention to the e1, e2 convention

        """
        params_to_realize = []
        for comp in self.components:
            comp_omega = getattr(self, comp).copy()
            profile = comp_omega.pop('profile') # e.g. 'PEMD'
            profile_params = comp_omega.keys()
            for param_name in profile_params:
                if (comp, param_name) not in params_to_exclude:
                    params_to_realize.append((comp, param_name))
        self.params_to_realize = params_to_realize

    def _define_kwargs_model(self):
        """Define the dictionary of profile list for each component

        """
        self.kwargs_model = {'lens_model_list': [self.lens_mass.profile, self.external_shear.profile],
                'source_light_model_list': [self.src_light.profile],
               #'point_source_model_list' : ['LENSED_POSITION']
            }
        if 'lens_light' in self.components:
            self.kwargs_model['lens_light_model_list'] = [self.lens_light.profile]
        if 'agn_light' in self.components:
            self.kwargs_model['point_source_model_list'] = ['SOURCE_POSITION']

    def set_comps_qphi_to_e1e2(self):
        comps_qphi_to_e1e2 = []
        for comp in self.components:
            comp_omega = getattr(self, comp).copy()
            profile = comp_omega.pop('profile') # e.g. 'PEMD'
            profile_params = comp_omega.keys()
            if ('e1' in self.params[profile]) and ((comp, 'e1') not in self.params_to_realize):
                comps_qphi_to_e1e2.append(comp)
        self.comps_qphi_to_e1e2 = comps_qphi_to_e1e2

    def _set_required_parameters(self):
        """Defines a dictionary of the list of parameters (value) corresponding to each profile (key).

        The parameter names follow the lenstronomy convention.
        The dictionary will be updated as more profiles are supported.

        """
        params = dict(PEMD=['center_x', 'center_y', 'gamma', 'theta_E', 'e1', 'e2'],
                          SHEAR_GAMMA_PSI=['gamma_ext', 'psi_ext'],
                          SERSIC_ELLIPSE=['magnitude', 'center_x', 'center_y', 'n_sersic', 'R_sersic', 'e1', 'e2'],
                          LENSED_POSITION=['magnitude'],
                          SOURCE_POSITION=['ra_source', 'dec_source', 'magnitude'],)
        setattr(self, 'params', params)

    def _raise_config_error(self, missing_key, parent_config_key, bnn_prior_class):
        """Convenience function for raising errors related to config values

        """
        raise ValueError("{:s} must be specified in the config inside {:s} for {:s}".format(missing_key, parent_config_key, bnn_prior_class))

    def sample_param(self, hyperparams):
        """Assigns a sampling distribution

        """
        hyperparams = hyperparams.copy()
        dist = hyperparams.pop('dist')
        return getattr(baobab.distributions, 'sample_{:s}'.format(dist))(**hyperparams)

    def eval_param_pdf(self, eval_at, hyperparams):
        """Assigns and evaluates the PDF 

        """
        hyperparams = hyperparams.copy()
        dist = hyperparams.pop('dist')
        return getattr(baobab.distributions, 'eval_{:s}_pdf'.format(dist))(eval_at, **hyperparams)

    @abstractmethod
    def sample(self):
        """Gets kwargs of sampled parameters to be passed to lenstronomy

        Overridden by subclasses.

        """
        return NotImplemented