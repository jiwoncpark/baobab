import numpy as np
import scipy.stats as stats
from astropy.cosmology import wCDM
import astropy.units as u
import lenstronomy.Util.param_util as param_util
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from .base_bnn_prior import BaseBNNPrior
from . import models

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
            dVol_dz = self.cosmo.differential_comoving_volume(z_grid).value
            dVol_dz_normed = dVol_dz/np.sum(dVol_dz)
            sampled_z = np.random.choice(z_grid, 2, replace=True, p=dVol_dz_normed)
            z_lens = np.min(sampled_z)
            z_src = np.max(sampled_z)
        else:
            raise NotImplementedError
        return z_lens, z_src

    def sample_velocity_dispersion(self, vel_disp_config):
        """ Sample velocity dispersion from the config-specified model,
        on a grid with the range and resolution specified in the config

        Parameters
        ----------
        vel_disp_config : dict
            Copy of cfg.bnn_omega.kinematics.velocity_dispersion

        Returns
        -------
        float
            a realization of velocity dispersion

        """
        vel_disp_grid = np.arange(**vel_disp_config.grid)
        if vel_disp_config.model == 'CPV2007':
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
        """Get the lens apparent magnitude from the Faber-Jackson relation
        given the realized velocity dispersion, with some scatter

        Parameters
        ----------
        vel_disp : float
            the velocity dispersion in km/s

        Note
        ----
        Does not account for peculiar velocity or dust. K-correction is approximate and implicit,
        as the absolute magnitude is in the V-band (480nm ~ 650nm) and, for z ~ 2-3, this portion 
        of the SED roughly lands in the IR.

        Returns
        -------
        m_V : float
            the apparent magnitude in the IR

        """
        log_L_V = models.luminosity_from_faber_jackson(vel_disp)
        M_V_sol = 4.84
        # FIXME: Enter good model for dust?
        A_V = 0.0 # V-band dust attenuation along LOS

        # FIXME: I could grab some template SEDs and K-correct explicitly, accounting for band throughput
        # for IR WF F140W. Should I do this?
        dist_mod = self.cosmo.distmod(z_lens).value
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
        r_eff = R_eff * self.cosmo.arcsec_per_kpc_comoving(z_lens).value # in arcsec
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
        gamma_with_scatter = models.gamma_from_size_relation(R_eff)
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
        hyperparams = dict(
                           dist='generalized_normal',
                           mu=np.pi,
                           alpha=np.pi,
                           p=10.0,
                           lower=0.0,
                           upper=2.0*np.pi,
                           )
        phi = self.sample_param(hyperparams)
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        return e1, e2

    def get_src_absolute_magnitude(self, z_src):
        """Sample the UV absolute magnitude from the luminosity function for the given redshift
        and convert into apparent magnitude

        Parameters
        ----------
        z_src : float
            the source redshift

        Returns
        -------
        float
            the absolute magnitude at 1500A

        """
        M_grid = np.arange(-23.0, -17.8, 0.2)
        nM_dM1500 = models.redshift_binned_luminosity_function(z_src, M_grid)
        nM_dM1500_normed = nM_dM1500/np.sum(nM_dM1500)
        M1500_src = np.random.choice(M_grid, None, replace=True, p=nM_dM1500_normed)
        return M1500_src

    def get_src_apparent_magnitude(self, z_src, M_src):
        """Convert the souce absolute magnitude into apparent magnitude

        Parameters
        ----------
        z_src : float
            the source redshift
        M_src : float
            the source absolute magnitude

        Note
        ----
        Does not account for peculiar velocity or dust. K-correction is approximate and implicit,
        as the absolute magnitude is at 150nm and, for z ~ 5-9, this portion 
        of the SED roughly lands in the IR.

        Returns
        -------
        float
            the apparent magnitude in the IR

        """
        dust = 0.0
        dist_mod = self.cosmo.distmod(z_src).value
        m_src = M_src + dist_mod - dust
        return m_src

    def get_src_size(self, z_src, M_V_src):
        """Get the effective radius of the source from its empirical relation with V-band absolute
        magnitude and redshift

        Parameters
        ----------
        M_V_src : float
            V-band absolute magnitude of the source
        z_src : float
            source redshift

        Returns
        -------
        tuple
            tuple of the effective radius in kpc and arcsec

        """
        R_eff = models.size_from_luminosity_and_redshift_relation(z_src, M_V_src)
        r_eff = R_eff * self.cosmo.arcsec_per_kpc_comoving(z_src).value # in arcsec
        return R_eff, r_eff

    def get_src_light_ellipticity(self):
        """Sample the source light ellipticity

        Returns
        -------
        tuple
            tuple of floats e1, e2

        """
        q = models.axis_ratio_disklike()
        # Approximately uniform in ellipticity angle
        hyperparams = dict(
                           dist='generalized_normal',
                           mu=np.pi,
                           alpha=np.pi,
                           p=10.0,
                           lower=0.0,
                           upper=2.0*np.pi,
                           )
        phi = self.sample_param(hyperparams)
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
        z_lens, z_src = self.sample_redshifts(redshifts_config=self.redshift)
        # Sample lens_mass and lens_light parameters
        vel_disp_iso = self.sample_velocity_dispersion(vel_disp_config=self.kinematics.velocity_dispersion)
        theta_E = self.get_theta_E_SIS(vel_disp_iso, z_lens, z_src)
        apmag_lens = self.get_lens_apparent_magnitude(vel_disp_iso, z_lens)
        R_eff_lens, r_eff_lens = self.get_lens_size(vel_disp_iso, z_lens, apmag_lens)
        gamma = self.get_gamma(R_eff_lens)
        lens_light_e1, lens_light_e2 = self.get_lens_light_ellipticity(vel_disp_iso)
        kwargs['lens_mass'] = dict(
                                   theta_E=theta_E,
                                   gamma=gamma,
                                   )
        kwargs['lens_light'] = dict(
                                    magnitude=apmag_lens,
                                    R_sersic=r_eff_lens,
                                    e1=lens_light_e1,
                                    e2=lens_light_e2,
                                    )
        kwargs['external_shear'] = {}

        # Sample src_light parameters
        abmag_src = self.get_src_absolute_magnitude(z_src)
        apmag_src = self.get_src_apparent_magnitude(z_src, abmag_src)
        R_eff_src, r_eff_src = self.get_src_size(z_src, abmag_src)
        src_light_e1, src_light_e2 = self.get_src_light_ellipticity()
        kwargs['src_light'] = dict(
                                   magnitude=apmag_src,
                                   R_sersic=r_eff_src,
                                   e1=src_light_e1,
                                   e2=src_light_e2,
                                   )
        # Sample AGN_light parameters
        kwargs['agn_light'] = {}

        # Miscellaneous other parameters to export
        kwargs['misc'] = dict(
                              z_lens=z_lens,
                              z_src=z_src,
                              vel_disp_iso=vel_disp_iso,
                              lens_light_R_eff=R_eff_lens,
                              src_light_R_eff=R_eff_src,
                              src_light_abmag=abmag_src,
                              )

        # Sample remaining parameters, not constrained by the above empirical relations,
        # independently from their (marginally) diagonal BNN prior
        for comp in self.components: # e.g. 'lens mass'
            comp_omega = getattr(self, comp).copy() # e.g. self.lens_mass
            profile = comp_omega.pop('profile') # e.g. 'SPEMD'
            profile_params = comp_omega.keys()
            for param_name in profile_params: # e.g. 'theta_E'
                if param_name not in kwargs[comp]:
                    hyperparams = comp_omega[param_name].copy()
                    kwargs[comp][param_name] = self.sample_param(hyperparams)

        # Source pos is defined wrt the lens pos
        kwargs['src_light']['center_x'] += kwargs['lens_mass']['center_x']
        kwargs['src_light']['center_y'] += kwargs['lens_mass']['center_y']

        if 'lens_light' in self.components:
            # Lens light shares center with lens mass
            kwargs['lens_light']['center_x'] = kwargs['lens_mass']['center_x']
            kwargs['lens_light']['center_y'] = kwargs['lens_mass']['center_y']

        return kwargs
