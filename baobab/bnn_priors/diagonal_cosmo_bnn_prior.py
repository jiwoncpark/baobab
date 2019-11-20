import numpy as np
import scipy.stats as stats
import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.Analysis.lens_properties import LensProp
from .diagonal_bnn_prior import DiagonalBNNPrior
from .base_cosmo_bnn_prior import BaseCosmoBNNPrior

class DiagonalCosmoBNNPrior(DiagonalBNNPrior, BaseCosmoBNNPrior):
    """BNN prior with independent parameters

    Note
    ----
    This BNNPrior is cosmology-aware. For a version that's not tailored to H0 inference, see `DiagonalBNNPrior`.

    """
    def __init__(self, bnn_omega, components):
        """
        Note
        ----
        The dictionary attributes are copies of the config corresponding to each component.
        The number of attributes depends on the number of components.

        Attributes
        ----------
        bnn_omega : dict
            copy of `cfg.bnn_omega`
        components : list
            list of components, e.g. `lens_mass`

        """
        DiagonalBNNPrior.__init__(self, bnn_omega, components)
        BaseCosmoBNNPrior.__init__(self, bnn_omega)
        self.params_to_exclude = []
        self.set_params_list(self.params_to_exclude)
        self.set_comps_qphi_to_e1e2()

    def sample(self):
        kwargs = DiagonalBNNPrior.sample(self)
        H0 = self.cosmology.H0
        z_lens, z_src = self.sample_redshifts(self.redshift.copy())
        kappa_ext = self.sample_param(self.LOS.kappa_ext.copy())
        lens_prop = LensProp(z_lens, z_src, self.kwargs_model, cosmo=self.cosmo)
        kwargs_lens_mass = dict(
                           theta_E=kwargs['lens_mass']['theta_E'],
                           gamma=kwargs['lens_mass']['gamma'],
                           center_x=kwargs['lens_mass']['center_x'],
                           center_y=kwargs['lens_mass']['center_y'],
                           e1=kwargs['lens_mass']['e1'],
                           e2=kwargs['lens_mass']['e2'],
                           )
        kwargs_ext_shear = dict(
                                gamma_ext=kwargs['external_shear']['gamma_ext'],
                                psi_ext=kwargs['external_shear']['psi_ext'],
                                )
        kwargs_lens = [kwargs_lens_mass, kwargs_ext_shear] # FIXME: hardcoded for SPEMD
        kwargs_ps = [dict(
                         ra_source=kwargs['src_light']['center_x'],
                         dec_source=kwargs['src_light']['center_y']
                         )]
        true_td = lens_prop.time_delays(kwargs_lens, kwargs_ps, kappa_ext=kappa_ext)
        measured_td = true_td + np.random.randn()*self.time_delays.error_sigma
        true_vd = lens_prop.velocity_dispersion(kwargs_lens, 
                                                    r_eff=kwargs['lens_light']['R_sersic'], 
                                                    R_slit=self.kinematics.aperture_size_x, 
                                                    dR_slit=self.kinematics.aperture_size_y, 
                                                    psf_fwhm=self.kinematics.psf_fwhm_eff, 
                                                    aniso_param=self.kinematics.aniso_param, 
                                                    num_evaluate=self.kinematics.num_evaluate, 
                                                    kappa_ext=kappa_ext)
        measured_vd = true_vd + true_vd*np.random.randn()*self.kinematics.vel_disp_frac_err_sigma
        kwargs['misc'] = dict(
                              z_lens=z_lens,
                              z_src=z_src,
                              measured_vd=measured_vd,
                              true_vd=true_vd,
                              measured_td=measured_td,
                              true_td=true_td,
                              kappa_ext=kappa_ext,
                              H0=H0,
                              )
        return kwargs