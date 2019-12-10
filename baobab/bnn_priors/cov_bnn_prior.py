import numpy as np
from addict import Dict
import lenstronomy.Util.param_util as param_util
from .base_bnn_prior import BaseBNNPrior
from baobab.distributions import sample_multivar_normal

class CovBNNPrior(BaseBNNPrior):
    """BNN prior with marginally covariant parameters

    Note
    ----
    This BNNPrior is cosmology-agnostic. For a version that's useful for H0 inference, see `CovCosmoBNNPrior`.

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
        BaseBNNPrior.__init__(self, bnn_omega, components)
        
        if 'cov_info' not in bnn_omega:
            raise self._raise_config_error('cov_info', 'bnn_omega', cls.__name__)
        self._check_cov_info_validity(bnn_omega['cov_info'])

        self.params_to_exclude = self.cov_info['cov_params_list']
        self.set_params_list(self.params_to_exclude)
        self.set_comps_qphi_to_e1e2()

    def _check_cov_info_validity(self, cov_info):
        """Checks whether the information passed into cov_info is valid.

        """
        n_cov_params = len(cov_info['cov_params_list'])
        cov_omega = cov_info['cov_omega']
        if len(cov_omega['mu']) != n_cov_params:
            raise ValueError("mu value in cov_omega should have same length as number of cov params in cov_params_list, {:d}, but instead found {:d}".format(n_cov_params, len(cov_omega['mu'])))
        if cov_omega['is_log'] is not None:
            if len(cov_omega['is_log']) != n_cov_params:
                raise ValueError("is_log value in cov_omega should have same length as number of cov params in cov_params_list, {:d}, but instead found {:d}".format(n_cov_params, len(cov_omega['is_log'])))
        if not np.array_equal(np.array(cov_omega['cov_mat']).shape, [n_cov_params, n_cov_params]):
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
        # Initialize nested dictionary of kwargs
        kwargs = Dict()

        # Realize samples
        for comp, param_name in self.params_to_realize:
            hyperparams = getattr(self, comp)[param_name].copy()
            kwargs[comp][param_name] = self.sample_param(hyperparams)

        # Fill in sampled values of covariant parameters
        cov_sample = sample_multivar_normal(**self.cov_info['cov_omega'])
        for i, (comp, param_name) in enumerate(self.cov_info['cov_params_list']):
            kwargs[comp][param_name] = cov_sample[i]

        # Convert any q, phi into e1, e2 as required by lenstronomy
        for comp in self.comps_qphi_to_e1e2: # e.g. 'lens_mass'
            q = kwargs[comp].pop('q')
            phi = kwargs[comp].pop('phi')
            e1, e2 = param_util.phi_q2_ellipticity(phi, q)
            kwargs[comp]['e1'] = e1
            kwargs[comp]['e2'] = e2
                
        # Source pos is defined wrt the lens pos
        kwargs['src_light']['center_x'] += kwargs['lens_mass']['center_x']
        kwargs['src_light']['center_y'] += kwargs['lens_mass']['center_y']

        # Ext shear is defined wrt the lens center
        kwargs['external_shear']['ra_0'] = kwargs['lens_mass']['center_x']
        kwargs['external_shear']['dec_0'] = kwargs['lens_mass']['center_y']

        if 'lens_light' in self.components:
            # Lens light shares center with lens mass
            kwargs['lens_light']['center_x'] = kwargs['lens_mass']['center_x']
            kwargs['lens_light']['center_y'] = kwargs['lens_mass']['center_y']
        return kwargs


