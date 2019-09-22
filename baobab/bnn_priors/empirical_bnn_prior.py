import numpy as np
import scipy.stats as stats
from astropy.cosmology import wCDM
import astropy.units as u
import lenstronomy.Util.param_util as param_util
from .base_bnn_prior import BaseBNNPrior
import baobab.models as models

class EmpiricalBNNPrior(BaseBNNPrior):
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
        super(EmpiricalBNNPrior, self).__init__()

        self.components = components
        self._check_empirical_omega_validity(bnn_omega)
        for comp in bnn_omega:
            setattr(self, comp, bnn_omega[comp])

        self.define_cosmology(self.cosmology)

    def _check_empirical_omega_validity(self, bnn_omega):
        """Check whether the config file specified the hyperparameters for all the fields
        required for `EmpiricalBNNPrior`, e.g. cosmology, redshift, galaxy kinematics

        """
        required_keys = ['cosmology', 'redshift', 'kinematics']
        for possible_missing_key in required_keys:
            if possible_missing_key not in bnn_omega:
            self._raise_config_error(possible_missing_key, 'bnn_omega', cls.__name__)

    def define_cosmology(self, cosmology_config):
        """Define the cosmology based on `cfg.bnn_omega.cosmology`

        Parameters
        ----------
        cosmology_config : dict
            Copy of cfg.bnn_omega.cosmology

        Returns
        -------
        astropy.cosmology.wCDM object
            the cosmology with which to generate all the training samples

        """
        self.cosmo = wCDM(**cosmology_config)

    def sample_redshifts(self, redshifts_config):
        """Sample redshifts from the differential comoving volume,
        on a grid with the range and resolution specified in the config

        Parameters
        ----------
        redshifts_config : dict
            Copy of cfg.bnn_omega.redshift

        Returns
        -------
        tuple
            the tuple of floats that are the realized z_lens, z_src

        """
        z_grid = np.arange(**redshifts_config.grid)
        if redshifts_config.model == 'differential_comoving_volume':
            dVol_dz = self.cosmo.differential_comoving_volume(z_grid)
            dVol_dz_normed = dVol_dz/np.sum(dVol_dz)
            sampled_z = np.random.choice(z_grid, 2, replace=True, p=dVol_dz_normed)
            z_lens = np.min(sampled_z)
            z_src = np.max(sampled_z)
        else:
            raise NotImplementedError
        return z_lens, z_src

    def sample_velocity_dispersion(self, veldisp_config):
        """ Sample velocity dispersion from the config-specified model,
        on a grid with the range and resolution specified in the config

        Parameters
        ----------
        veldisp_config : dict
            Copy of cfg.bnn_omega.kinematics.velocity_dispersion

        Returns
        -------
        float
            a realization of velocity dispersion

        """
        vel_disp_grid = np.arange(**veldisp_config.grid)
        if veldisp_config.model == 'CPV2007':
            dn = models.velocity_dispersion_function_CPV2007(vel_disp_grid)
        else:
            raise NotImplementedError
        dn_normed = dn/np.sum(dn)
        sampled_vel_disp = np.random.choice(vel_disp_grid, None, replace=True, p=dn_normed)
        return sampled_vel_disp

    def get_theta_E_SIS(self, vel_disp_iso, z_lens, z_src):
        """Compute the Einstein radius for a given isotropic velocity dispersion
        assuming a singular isothermal sphere (SIS) mass profile

        Parameters
        ----------
        vel_disp_iso : float 
            isotropic velocity dispersion, or an approximation to it, in km/s
        z_lens : float
            the lens redshift

        z_src : float
            the source redshift

        Note
        ----
        The computation is purely analytic.
        .. math:: \theta_E = 4 \pi \frac{\sigma_V^2}{c^2} \frac{D_{ls}}{D_s}

        Returns
        -------
        float
            the Einstein radius for an SIS in arcsec

        """
        lens_cosmo = LensCosmo(z_lens, z_src, cosmo=self.cosmo)
        theta_E_SIS = lens_cosmo.sis_sigma_v2theta_E(vel_disp_iso)
        return theta_E_SIS

    def get_lens_apparent_magnitude(self, vel_disp, z_lens):
        """Get the lens V-band apparent magnitude from the Faber-Jackson relation
        given the realized velocity dispersion, with some scatter

        Parameters
        ----------
        vel_disp : float
            the velocity dispersion in km/s

        Note
        ----
        Does not account for peculiar velocity.

        Returns
        -------
        m_V : float
            the V-band apparent magnitude

        """
        log_L_V = models.luminosity_from_faber_jackson(vel_disp)
        M_V_sol = 4.84
        A_V = 0.2 # V-band dust attenuation along LOS

        dist_mod = self.cosmo.distmod(z_lens)
        M_V = -2.5 * log_L_V + M_V_sol
        m_V = dist_mod - A_V
        return m_V

    def get_lens_size(self, vel_disp, z_lens, m_V):
        """Get the lens V-band efefctive radius from the Fundamental Plane relation
        given the realized velocity dispersion and apparent magnitude, with some scatter

        Parameters
        ----------
        vel_disp : float
            the velocity dispersion in km/s
        z_lens : float
            redshift
        m_V : float
            V-band apparent magnitude

        Returns
        -------
        tuple
            the effective radius in kpc and arcsec

        """
        R_eff = models.size_from_fundamental_plane(vel_disp, m_V) # in kpc
        r_eff = R_eff * self.cosmo.arcsec_per_kpc_comoving(z) # in arcsec
        return R_eff, r_eff

    def get_gamma(self, R_eff):
        """Get the power-law slope of the mass profile using the fit derived from the SLACS
        sample

        Parameters
        ----------
        R_eff : float
            effective radius of the lens light in kpc

        Returns
        -------
        float
            gamma with random scatter from propgated fit errors and intrinsic scatter

        """
        gamma_with_scatter = models.gamma_from_size_correlation(R_eff)
        return gamma_with_scatter

    def get_lens_light_ellipticity(self, vel_disp):
        """Get the lens light ellipticity from a reasonable distribution agreeing with the SDSS data

        Parameters
        ----------
        vel_disp : float
            velocity dispersion in km/s

        Returns
        -------
        tuple
            tuple of floats e1, e2

        """
        q = models.axis_ratio_from_SDSS(vel_disp)
        # Approximately uniform in ellipticity angle
        phi = self.sample_param(dist='generalized_normal',
                                mu=np.pi,
                                alpha=np.pi,
                                p=10.0,
                                lower=0.0,
                                upper=2.0*np.pi)
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        return e1, e2

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
        z_lens, z_src = self.sample_redshifts(self.redshift)
        # Sample lens_mass and lens_light parameters
        vel_disp_iso = self.sample_velocity_dispersion(self.kinematics.velocity_dispersion)
        theta_E = self.get_theta_E_SIS(vel_disp_iso=vel_disp_iso, z_lens=z_lens, z_src=z_src)
        mag_lens = self.get_lens_apparent_magnitude(vel_disp=vel_disp_iso, z_lens=z_lens)
        R_eff, r_eff = self.get_lens_size(vel_disp=vel_disp_iso, z_lens=z_lens, m_V=mag_lens)
        gamma = self.get_gamma(R_eff)
        lens_light_e1, lens_light_e2 = self.get_lens_light_ellipticity(vel_disp_iso)
        kwargs['lens_mass'] = dict(
                                   theta_E=theta_E,
                                   gamma=gamma,
                                   )
        kwargs['lens_light'] = dict(
                                    magnitude=mag_lens,
                                    R_sersic=r_eff,
                                    e1=lens_light_e1,
                                    e2=lens_light_e2,
                                    )
        # Sample src_light parameters
        
        # draw source luminosity in redshift bin
        # draw effective radius like LensPop

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

        # Source pos is defined wrt the lens pos
        kwargs['src_light']['center_x'] += kwargs['lens_mass']['center_x']
        kwargs['src_light']['center_y'] += kwargs['lens_mass']['center_y']

        if 'lens_light' in self.components:
            # Lens light shares center with lens mass
            kwargs['lens_light']['center_x'] = kwargs['lens_mass']['center_x']
            kwargs['lens_light']['center_y'] = kwargs['lens_mass']['center_y']
        return kwargs


