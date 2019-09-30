import warnings
import numpy as np
import scipy.stats as stats
import lenstronomy.Util.param_util as param_util
from .base_bnn_prior import BaseBNNPrior

class CovBNNPrior(BaseBNNPrior):
    """BNN prior with marginally covariant parameters

    """
    def __init__(self, bnn_omega, components):
        """
        Note
        ----
        The dictionary attributes are copies of the config corresponding to each component.
        The number of attributes depends on the number of components.

        Attributes
        ----------
        components : list
            list of components, e.g. `lens_mass`
        lens_mass : dict
            profile type and parameters of the lens mass
        src_light : dict
            profile type and parameters of the source light
        """
        super(CovBNNPrior, self).__init__()
        if 'cov_info' not in bnn_omega:
            raise self._raise_config_error('cov_info', 'bnn_omega', cls.__name__)
        
        self.components = components
        self._check_cov_info_validity(bnn_omega['cov_info'])
        
        for comp in bnn_omega: 
            # e.g. self.lens_mass = cfg.bnn_omega.lens_mass
            setattr(self, comp, bnn_omega[comp])

    def _check_cov_info_validity(self, cov_info):
        """Checks whether the information passed into cov_info is valid.

        """
        if len(set(cov_info['cov_params_list']) - set(self.components)) != 0:
            warnings.warn("You specified covariance between parameters for profiles not in components list.")
           
        n_cov_params = len(cov_info['cov_params_list'])
        cov_omega = cov_info['cov_omega']
        if len(cov_omega['mu']) != n_cov_params:
            raise ValueError("mu value in cov_omega should have same length as number of cov params in cov_params_list, {:d}, but instead found {:d}".format(n_cov_params, len(cov_omega['mu'])))
        if cov_omega['is_log'] is not None:
            if len(cov_omega['is_log']) != n_cov_params:
                raise ValueError("is_log value in cov_omega should have same length as number of cov params in cov_params_list, {:d}, but instead found {:d}".format(n_cov_params, len(cov_omega['is_log'])))
        if not np.array_equal(cov_omega['cov_mat'].shape, [n_cov_params, n_cov_params]):
            raise ValueError("cov_mat value in cov_omega should have shape [n_cov_params, n_cov_params]")

    def sample(self):
        """Gets kwargs of sampled parameters to be passed to lenstronomy

        Returns
        -------
        dict
            dictionary of config-specified components (e.g. lens mass), itself
            a dictionary of sampled parameters corresponding to the config-specified
            profile of that component

            """
        kwargs = {}
        for comp in self.components: # e.g. 'lens mass'
            kwargs[comp] = {}
            comp_omega = getattr(self, comp).copy() # e.g. self.lens_mass
            profile = comp_omega.pop('profile') # e.g. 'SPEMD'
            profile_params = comp_omega.keys()
            for param_name in profile_params: # e.g. 'theta_E'
                if (comp, param_name) not in self.cov_info['cov_params_list']:
                    hyperparams = comp_omega[param_name].copy()
                    kwargs[comp][param_name] = self.sample_param(hyperparams)

        # Fill in sampled values of covariant parameters
        cov_sample = self.sample_multivar_normal(**self.cov_info['cov_omega'])
        for i, (comp, param_name) in enumerate(self.cov_info['cov_params_list']):
            kwargs[comp][param_name] = cov_sample[i]

        # Convert any q, phi into e1, e2 as required by lenstronomy
        for comp in self.components: # e.g. 'lens_mass'
            comp_omega = getattr(self, comp).copy() # e.g. self.lens_mass
            profile = comp_omega.pop('profile') # e.g. 'SPEMD'
            if ('e1' in self.params[profile]) and ('e1' not in kwargs[comp]):
                q = kwargs[comp].pop('q')
                phi = kwargs[comp].pop('phi')
                e1, e2 = param_util.phi_q2_ellipticity(phi, q)
                kwargs[comp]['e1'] = e1
                kwargs[comp]['e2'] = e2
                
        # Source pos is defined wrt the lens pos
        kwargs['src_light']['center_x'] += kwargs['lens_mass']['center_x']
        kwargs['src_light']['center_y'] += kwargs['lens_mass']['center_y']

        if 'lens_light' in self.components:
            # Lens light shares center with lens mass
            kwargs['lens_light']['center_x'] = kwargs['lens_mass']['center_x']
            kwargs['lens_light']['center_y'] = kwargs['lens_mass']['center_y']
        return kwargs


