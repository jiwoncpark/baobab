import numpy as np
import scipy.stats as stats
import lenstronomy.Util.param_util as param_util
from .base_bnn_prior import BaseBNNPrior

class DiagonalBNNPrior(BaseBNNPrior):
    """BNN prior with independent parameters

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
        super(DiagonalBNNPrior, self).__init__()
        self.components = components
        for comp in bnn_omega:
            if comp in self.components:
                # e.g. self.lens_mass = cfg.bnn_omega.lens_mass
                setattr(self, comp, bnn_omega[comp])

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
                hyperparams = comp_omega[param_name].copy()
                kwargs[comp][param_name] = self.sample_param(hyperparams)

        # Source pos is defined wrt the lens pos
        kwargs['src_light']['center_x'] += kwargs['lens_mass']['center_x']
        kwargs['src_light']['center_y'] += kwargs['lens_mass']['center_y']

        # Lens light shares center with lens mass
        kwargs['lens_light']['center_x'] = kwargs['lens_mass']['center_x']
        kwargs['lens_light']['center_y'] = kwargs['lens_mass']['center_y']
        return kwargs


