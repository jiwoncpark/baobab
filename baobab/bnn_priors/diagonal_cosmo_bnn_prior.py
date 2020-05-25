import numpy as np
from lenstronomy.Analysis.td_cosmography import TDCosmography
from .diagonal_bnn_prior import DiagonalBNNPrior
from .base_cosmo_bnn_prior import BaseCosmoBNNPrior
import baobab.sim_utils.kinematics_utils as kinematics_utils

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
        if self.kinematics.calculate_vel_disp or self.time_delays.calculate_time_delays:
            self.get_cosmography_observables = True
        else:
            self.get_cosmography_observables = False
        self.get_velocity_dispersion = getattr(kinematics_utils, 'velocity_dispersion_analytic') if self.kinematics.anisotropy_model == 'analytic' else getattr(kinematics_utils, 'velocity_dispersion_numerical')

    def get_cosmo_observables(self, kwargs, z_lens, z_src, kappa_ext):
        """Calculate the central estimates of the observables for cosmography, i.e. the velocity dispersion and time delays, with and without noise realization

        Parameters
        ----------
        kwargs : dict
            the realized kwargs
        z_lens : float
        z_src : float
        kappa_ext : float

        Returns
        -------
        dict
            the computed central estimates of velocity dispersion and time delays with and without noise realization

        """
        td_cosmo = TDCosmography(z_lens, z_src, self.kwargs_model, cosmo_fiducial=self.cosmo)
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
                                ra_0=kwargs['external_shear']['ra_0'],
                                dec_0=kwargs['external_shear']['dec_0'],
                                )
        kwargs_lens = [kwargs_lens_mass, kwargs_ext_shear] # FIXME: hardcoded for SPEMD
        kwargs_lens_light = [kwargs['lens_light']]
        kwargs_ps = [dict(ra_source=kwargs['src_light']['center_x'],
                          dec_source=kwargs['src_light']['center_y'])]
        # Time delays
        if self.time_delays.calculate_time_delays:
            true_td, x_image, y_image = td_cosmo.time_delays(kwargs_lens, kwargs_ps, kappa_ext=kappa_ext)
        else:
            true_td = -1
        # Velocity dispersion
        if self.kinematics.calculate_vel_disp:
            true_vd = self.get_velocity_dispersion(
                                                   td_cosmo, 
                                                   kwargs_lens, 
                                                   kwargs_lens_light, 
                                                   self.kinematics.kwargs_anisotropy, 
                                                   self.kinematics.kwargs_aperture, 
                                                   self.kinematics.kwargs_psf, 
                                                   self.kinematics.anisotropy_model, 
                                                   kwargs['lens_light']['R_sersic'],
                                                   self.kinematics.kwargs_numerics,
                                                   kappa_ext,
                                                   )
        else:
            true_vd = -1
        obs = dict(true_td=true_td.tolist(),
                   true_vd=true_vd,
                   x_image=x_image,
                   y_image=y_image
                   )
        return obs

    def sample(self):
        kwargs = DiagonalBNNPrior.sample(self)
        H0 = self.cosmology.H0
        z_lens, z_src = self.sample_redshifts(self.redshift.copy())
        kappa_ext = self.sample_param(self.LOS.kappa_ext.copy())
        kwargs['misc'] = dict(
                             z_lens=z_lens,
                             z_src=z_src,
                             kappa_ext=kappa_ext,
                             H0=H0,
                             )
        if self.get_cosmography_observables:
            cosmo_obs = self.get_cosmo_observables(kwargs, z_lens, z_src, kappa_ext)
            kwargs['misc'].update(cosmo_obs)
        return kwargs