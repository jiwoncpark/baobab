import numpy as np
import scipy.stats as stats
import lenstronomy.Util.param_util as param_util
from .diagonal_bnn_prior import DiagonalBNNPrior
from .base_cosmo_bnn_prior import BaseCosmoBNNPrior

class Rung1BNNPrior(DiagonalBNNPrior, BaseCosmoBNNPrior):
    """BNN prior identical to the one used in TDLMC Rung 1

    Note
    ----
    This BNNPrior is cosmology-aware. 

    Parameters
    ----------
    bnn_omega : dict
        copy of `cfg.bnn_omega`
    components : list
        list of components, e.g. `lens_mass`

    """
    def __init__(self, bnn_omega, components):
        DiagonalBNNPrior.__init__(self, bnn_omega, components)
        BaseCosmoBNNPrior.__init__(self, bnn_omega)
        self.exclude_from_independent_sampling = [('lens_mass', 'theta_E'), ('lens_light', 'R_sersic'), ('src_light', 'magnitude')]

    def sample(self):
        z_lens, z_src = self.sample_redshifts(self.redshift)
        kappa_ext = self.sample_param(self.LOS.kappa_ext)
        vel_disp = self.sample_param(self.kinematics.vel_disp)

        kwargs = {}
        for comp in self.components: # e.g. 'lens mass'
            kwargs[comp] = {}
            comp_omega = getattr(self, comp).copy() # e.g. self.lens_mass
            profile = comp_omega.pop('profile') # e.g. 'SPEMD'
            profile_params = comp_omega.keys()
            for param_name in profile_params: # e.g. 'theta_E'
                hyperparams = comp_omega[param_name].copy()
                kwargs[comp][param_name] = self.sample_param(hyperparams)

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


        kwargs['misc'] = dict(
                              z_lens=z_lens,
                              z_src=z_src,
                              measured_vd=measured_vd,
                              true_vd=true_vd,
                              measured_td=measured_td,
                              true_td=true_td,
                              kappa_ext=kappa_ext,
                              H0=true_H0,
                              )
        return kwargs