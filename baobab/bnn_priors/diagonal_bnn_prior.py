import numpy as np
import scipy.stats as stats
from addict import Dict
import lenstronomy.Util.param_util as param_util
from .base_bnn_prior import BaseBNNPrior

class DiagonalBNNPrior(BaseBNNPrior):
    """BNN prior with independent parameters

    Note
    ----
    This BNNPrior is cosmology-agnostic. For a version that's useful for H0 inference, see `DiagonalCosmoBNNPrior`.

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
        self.params_to_exclude = []
        self.set_params_list(self.params_to_exclude)
        self.set_comps_qphi_to_e1e2()

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


