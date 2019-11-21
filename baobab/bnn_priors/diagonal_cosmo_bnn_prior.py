import numpy as np
from lenstronomy.Analysis.td_cosmography import TDCosmography
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
        self.get_velocity_dispersion = getattr(self, 'velocity_dispersion_analytic') if self.kinematics.anisotropy_model == 'analytic' else getattr(self, 'velocity_dispersion_numerical')

    def velocity_dispersion_analytic(self, td_cosmo_object, kwargs_lens, kwargs_lens_light, kwargs_anisotropy, kwargs_aperture, kwargs_psf, anisotropy_model, r_eff, kwargs_numerics, kappa_ext):
        """Get the LOS velocity dispersion of the lens within a square slit of given width and length and seeing with the given FWHM. The computation is analytic as it assumes a Hernquist light profiel and a spherical power-law lens model at the first position.

        Parameters
        ----------
        td_cosmo_object : `lenstronomy.Analysis.TDCosmography` object
            tool with which to compute the velocity dispersion
        kwargs_lens : list of dict
            lens mass parameters
        kwargs_lens_light : list of dict
            lens light parameters
        kwargs_anisotropy : dict
            anisotropy parameters such as `r_ani`
        kwargs_aperture : dict
            aperture geometry
        kwargs_psf : dict
            seeing conditions
        anisotropy_model : str
            `analytic` if using this module, else the model to evaluate numerically, e.g. `OsipkovMerritt`
        r_eff : float
            rough estimate of the half-light radius of the lens light
        kwargs_numerics : dict
            numerical solver config

        Returns
        -------
        float
            the sigma of the velocity dispersion
            
        """

        module = getattr(td_cosmo_object, 'velocity_dispersion_analytical')
        vel_disp = module(
                          theta_E=kwargs_lens[0]['theta_E'],
                          gamma=kwargs_lens[0]['gamma'],
                          r_ani=kwargs_anisotropy['r_ani'],
                          r_eff=r_eff,
                          kwargs_aperture=kwargs_aperture,
                          kwargs_psf=kwargs_psf,
                          num_evaluate=kwargs_numerics['sampling_number'],
                          kappa_ext=kappa_ext,
                          )
        return vel_disp

    def velocity_dispersion_numerical(self, td_cosmo_object, kwargs_lens, kwargs_lens_light, kwargs_anisotropy, kwargs_aperture, kwargs_psf, anisotropy_model, r_eff, kwargs_numerics, kappa_ext):
        """Get the velocity dispersion using a numerical model

        See `velocity_dispersion_analytic` for the parameter description.

        """
        module = getattr(td_cosmo_object, 'velocity_dispersion_numerical')
        vel_disp = module(
                          kwargs_lens=kwargs_lens,
                          kwargs_lens_light=kwargs_lens_light,
                          kwargs_anisotropy=kwargs_anisotropy,
                          kwargs_aperture=kwargs_aperture,
                          kwargs_psf=kwargs_psf,
                          MGE_light=False,
                          kwargs_mge_light=False,
                          MGE_mass=False,
                          kwargs_mge_mass=False,
                          Hernquist_approx=False,
                          anisotropy_model=anisotropy_model,
                          r_eff=r_eff,
                          kwargs_numerics=kwargs_numerics,
                          kappa_ext=kappa_ext,
                          )
        return vel_disp

    def sample(self):
        kwargs = DiagonalBNNPrior.sample(self)
        H0 = self.cosmology.H0
        z_lens, z_src = self.sample_redshifts(self.redshift.copy())
        kappa_ext = self.sample_param(self.LOS.kappa_ext.copy())
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
                                )
        kwargs_lens = [kwargs_lens_mass, kwargs_ext_shear] # FIXME: hardcoded for SPEMD
        kwargs_lens_light = [kwargs['lens_light']]
        kwargs_ps = [dict(
                         ra_source=kwargs['src_light']['center_x'],
                         dec_source=kwargs['src_light']['center_y']
                         )]
        true_td = td_cosmo.time_delays(kwargs_lens, kwargs_ps, kappa_ext=kappa_ext)
        measured_td = true_td + np.random.randn()*self.time_delays.error_sigma
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
                                               kappa_ext
                                               )
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