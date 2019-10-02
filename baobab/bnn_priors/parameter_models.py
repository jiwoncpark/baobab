import numpy as np
from scipy.special import gamma
import astropy.units as u

class FaberJackson:
	"""Represents the Faber-Jackson (FJ) relation between velocity dispersion and luminosity
	of elliptical galaxies.

	FJ is a projection of the Fundamental Plane (FP) relation.

	"""
	def __init__(self, slope=None, intercept=None, fit_data=None):
		"""
		Parameters
		----------
		slope : float
			linear slope of the log(L_V/L_solar) vs. log(vel_disp/(km/s)) relation
		intercept : float
			intercept of the log(L_V/L_solar) vs. log(vel_disp/(km/s)) relation, i.e.
			the value of log(L_V/L_solar) at vel_disp = 1 km/s.
		fit_data : float
			sample on which the slope and intercept were fit (one of ['ETGS']). Default: None

		"""
		if fit_data is None and (slope is None or intercept is None):
			raise ValueError("Either all the fit parameters or fit_data must be specified.")
		if not (fit_data is None or slope is None or intercept is None):
			raise ValueError("Cannot specify fit parameters when fit_data is specified.")

		self.slope = slope
		self.intercept = intercept
		if fit_data == 'ETGs':
			self._define_ETG_fit_params()
		else:
			raise NotImplementedError

	def _define_ETG_fit_params(self):
		"""Set the slope and intercept values fit on a sample of ETGs

		Note
		----
		The slope and intercept were read off from Fig 7 of [1]_.
		Values binned by magnitudes are available in [2]_.

		References
		----------
		.. [1] D’Onofrio, Mauro, et al. "On the Origin of the Fundamental Plane and Faber–Jackson Relations: Implications for the Star Formation Problem." The Astrophysical Journal 838.2 (2017): 163.

		.. [2] Nigoche-Netro, A., et al. "The Faber-Jackson relation for early-type galaxies: dependence on the magnitude range." Astronomy & Astrophysics 516 (2010): A96.

		"""
		self.slope = 2.0
		self.intercept = 5.8   

	def get_luminosity(self, vel_disp):
		"""Evaluate the V-band luminosity L_V expected from the FJ relation
		for a given velocity dispersion

		Parameters
		----------
		vel_disp : float
			the velocity dispersion in km/s

		Returns
		-------
		float
			log(L_V/L_solar)

		"""
		log_L_V = self.slope*np.log10(vel_disp) + self.intercept
		return log_L_V

class FundamentalPlane:
	"""Represents the Fundamental Plane (FP) relation between the velocity dispersion,
	luminosity, and effective radius for elliptical galaxies

	Luminosity is expressed as apparent magnitude in this form.

	"""
	def __init__(self, a=None, b=None, c=None, fit_data=None):
		"""
		Parameters
		----------
		a : float
			linear slope on the log velocity dispersion, log(vel_disp/(km/s))
		b : float
			linear slope on the V-band apparent magnitude, or m_V/mag
		c : float
			intercept, i.e. the log effective radius, or log(R_eff/kpc),
			when vel_disp = m_V = 0
		fit_data : str
			sample on which a, b, c were fit (one of ['SDSS']). Default: None

		"""
		if fit_data is None and (a is None or b is None or c is None):
			raise ValueError("Either all the fit parameters or fit_data must be specified.")
		if not (fit_data is None or a is None or b is None or c is None):
			raise ValueError("Cannot specify fit parameters when fit_data is specified.")

		self.a = a
		self.b = b
		self.c = c
		if fit_data == 'SDSS':
			self._define_SDSS_fit_params()
		else:
			raise NotImplementedError

	def _define_SDSS_fit_params(self):
		"""Set the parameters fit on SDSS DR4

		Note
		----
		The values of slope and intercept are taken from the r-band orthogonal fit
		on SDSS DR4. See Table 2 of [1]_.
		
		References
		----------
		.. [1] Hyde, Joseph B., and Mariangela Bernardi. 
		"The luminosity and stellar mass Fundamental Plane of early-type galaxies." 
		Monthly Notices of the Royal Astronomical Society 396.2 (2009): 1171-1185.

		"""
		self.a = 1.4335
		self.b = 0.3150 
		self.c = -8.8979

	def get_effective_radius(self, vel_disp, m_V):
		"""Evaluate the size expected from the FP relation
		for a given velocity dispersion and V-band apparent magnitude

		Parameters
		----------
		vel_disp : float
			the velocity dispersion in km/s
		m_V : float
			the apparent V-band magnitude

		Returns
		-------
		float
			the effective radius in kpc

		"""

		log_R_eff = self.a*np.log10(vel_disp) + self.b*m_V + self.c
		R_eff = 10**log_R_eff
		return R_eff

class FundamentalMassHyperplane:
	"""Represents bivariate relations (projections) within the Fundamental Mass Hyperplane (FMHP) relation 
	between the stellar mass, stellar mass density, effective radius, and velocity dispersion of massive ETGs.

	Only the relation between the power-law mass slope (gamma) and effective radius is currently supported.

	"""
	def __init__(self, a=None, b=None, delta_a=0.0, delta_b=0.0, intrinsic_scatter=0.0, fit_data=None):
		"""
		Parameters
		----------
		a : float
		the linear slope of the log(gamma) vs. log(R_eff/kpc) relation
		b : float
			the intercept of the log(gamma) vs. log(R_eff/kpc) relation, i.e.
			the value of log(gamma) at R_eff = 1 kpc
		delta_a : float
			1-sigma fit error on the slope. Default: 0
		delta_b : float
			1-sigma fit error on the intercept. Default: 0
		intrinsic_scatter : float
			1-sigma intrinsic scatter, i.e. error on the log(R_eff/kpc) measurements. Default: 0
		fit_data : str
			sample on which a, b were fit (one of ['SLACS']). Default: None

		"""
		if fit_data is None and (a is None or b is None):
			raise ValueError("Either all the fit parameters or fit_data must be specified.")
		if not (fit_data is None or a is None or b is None):
			raise ValueError("Cannot specify fit parameters when fit_data is specified.")

		self.a = a
		self.b = b
		self.delta_a = delta_a
		self.delta_b = delta_b
		self.intrinsic_scatter = intrinsic_scatter

		if fit_data == 'SLACS':
			self._define_SLACS_fit_params()
		else:
			raise NotImplementedError

	def _define_SLACS_fit_params(self):
		"""Set the parameters fit on the Sloan Lens Arcs Survey (SLACS) sample of 73 ETGs

		Note
		----
		See Table 4 of [1]_ for the fit values, taken from the empirical correlation derived 
		from the SLACS lens galaxy sample.

		References
		----------
		.. [1] Auger, M. W., et al. "The Sloan Lens ACS Survey. X. Stellar, dynamical, and total mass correlations of massive early-type galaxies." The Astrophysical Journal 724.1 (2010): 511.

		"""
		self.a = -0.41
		self.b = 0.39
		self.delta_a = 0.12
		self.delta_b = 0.10
		self.intrinsic_scatter = 0.14

	def get_gamma(self, R_eff):
		"""Evaluate the power-law slope of the mass profile from its power-law relation with effective radius

		Parameters
		----------
		R_eff : float
			the effective radius in kpc

		Returns
		-------
		float
			the power-law slope, gamma

		"""
		log_R_eff = np.log10(R_eff)
		gamma_minus_2 = log_R_eff*self.a + self.b
		gamma = gamma_minus_2 + 2.0
		gamma_sig = (self.intrinsic_scatter**2.0 + np.abs(log_R_eff)*self.delta_a**2.0 + self.delta_b**2.0)**0.5
		scatter = np.random.randn()*gamma_sig
		return gamma + scatter

class AxisRatioRayleigh:
	"""Represents various scaling relations that the axis ratio can follow with 
	quantities like velocity dispersion, when its PDF is assumed to be a Rayleigh distribution

	Only the relation with velocity dispersion is currently supported.

	"""
	def __init__(self, a=None, b=None, lower=0.2, fit_data=None):
		"""
		Parameters
		----------
		a : float
			linear slope of the scale vs. velocity dispersion relation, in (km/s)^-1, i.e.
			how much the velocity dispersion contributes to average flattening 
		b : float
			intercept of the scale vs. velocity dispersion relation, i.e.
			the mean flattening independent of velocity dispersion
		lower : float
			minimum allowed value of the axis ratio. Default: 0.2
		fit_data : str
			sample on which a, b were fit (one of ['SDSS']). Default: None

		"""
		if fit_data is None and (a is None or b is None):
			raise ValueError("Either all the fit parameters or fit_data must be specified.")
		if not (fit_data is None or a is None or b is None):
			raise ValueError("Cannot specify fit parameters when fit_data is specified.")

		self.a = a
		self.b = b
		self.lower = lower

		if fit_data == 'SDSS':
			self._define_SDSS_fit_params()
		else:
			raise NotImplementedError

	def _define_SDSS_fit_params(self):
		"""Set the parameters fit on the SDSS data

		Note
		----
		The shape of the distribution arises because more massive galaxies are closer to spherical than 
		less massive ones. The truncation excludes highly-flattened profiles. 
		The default fit values have been derived by [1]_ from the SDSS data. 

		References
		----------
		.. [1] Collett, Thomas E. "The population of galaxy–galaxy strong lenses in forthcoming optical imaging surveys." The Astrophysical Journal 811.1 (2015): 20.

		"""
		self.a = 5.7*1.e-4
		self.b = 0.38
		self.lower = 0.2

	def get_axis_ratio(self, vel_disp):
		"""Sample (one minus) the axis ratio of the lens galaxy from the Rayleigh distribution with scale
		that depends on velocity dispersion

		Parameters
		----------
		vel_disp : float
			velocity dispersion in km/s

		Returns
		-------
		float
			the axis ratio q

		"""
		scale = self.a*vel_disp + self.b
		q = 0.0
		while q < self.lower:
			q = 1.0 - np.random.rayleigh(scale, size=None)
		return q

def redshift_binned_luminosity_function(z, M_grid):
	"""Sample FUV absolute magnitude from the redshift-binned luminosity function

	Parameters
	----------
	z : float
		galaxy redshift
	M_grid : array-like
		grid of FUV absolute magnitudes at which to evaluate luminosity function

	Note
	----
	For z < 4, we use the Schechter function fits in Table 1 of [1]_ and,
	for 4 < z < 8, those in Table 4 of [2]_.
	z > 8 are binned into the z=8 bin. I might add high-redshift models, e.g. from [3]_.

	References
	----------
	.. [1] Arnouts, Stephane, et al. "The GALEX VIMOS-VLT Deep Survey* Measurement of the Evolution of the 1500 Å Luminosity Function." The Astrophysical Journal Letters 619.1 (2005): L43.

	.. [2] Finkelstein, Steven L., et al. "The evolution of the galaxy rest-frame ultraviolet luminosity function over the first two billion years." The Astrophysical Journal 810.1 (2015): 71.

	.. [3] Kawamata, Ryota, et al. "Size–Luminosity Relations and UV Luminosity Functions at z= 6–9 Simultaneously Derived from the Complete Hubble Frontier Fields Data." The Astrophysical Journal 855.1 (2018): 4.

	Returns
	-------
	array-like
		unnormalized function of the absolute magnitude at 1500A

	"""
	#prefactor = np.log(10)*phi_star # just normalization
	# Define redshift bins by right edge of bin
	z_bins = np.array([0.2, 0.4, 0.6, 0.8, 1.2, 2.25, 3.4, 4.5, 5.5, 6.5, 7.5, np.inf])
	alphas = np.array([-1.21, -1.19, -1.55, -1.60, -1.63, -1.49, -1.47, -1.56, -1.67, -2.02, -2.03, -2.36])
	M_stars = np.array([-18.05, -18.38, -19.49, -19.84, -20.11, -20.33, -21.08, -20.73, -20.81, -21.13, -21.03, -20.89])
	alpha = alphas[z < z_bins][0]
	M_star = M_stars[z < z_bins][0]

	# Note phi_star is ignored as normalization
	# Schechter kernel
	exponent = 10.0**(0.4*(M_star - M_grid))
	density = np.exp(-exponent) * exponent**(alpha + 1.0)
	return density

def size_from_luminosity_and_redshift_relation(z, M_V):
	"""Sample the effective radius of Lyman break galaxies from the relation with luminosity and redshift

	Parameters
	----------
	z : float
		galaxy redshift
	M_V : float
		V-band absolute magnitude

	Note
	----
	The relation and scatter agree with [1]_ and [2]_, which both show that size decreases
	with higher redshift. They have been used in LensPop ([3]_).

	References
	----------
	.. [1] Mosleh, Moein, et al. "The evolution of mass-size relation for Lyman break galaxies from z= 1 to z= 7." 	The Astrophysical Journal Letters 756.1 (2012): L12.

	.. [2] Huang, Kuang-Han, et al. "The bivariate size-luminosity relations for Lyman break galaxies at z∼ 4-5." The Astrophysical Journal 765.1 (2013): 68.

	.. [3] Collett, Thomas E. "The population of galaxy–galaxy strong lenses in forthcoming optical imaging surveys." The Astrophysical Journal 811.1 (2015): 20.

	Returns
	-------
	float
		a sampled effective radius in kpc

	"""
	log_R_eff = (M_V/-19.5)**-0.22 * ((1.0 + z)/5.0)**-1.2
	scatter = np.random.randn()*0.3
	log_R_eff += scatter
	R_eff = 10.0**log_R_eff
	return R_eff

class AGNLuminosityFunction:
	"""Redshift-binned AGN luminosity function parameterized as a double power-law

	"""

	def __init__(self, M_grid, z_bins=None, alphas=None, betas=None, M_stars=None, fit_data=None):
		"""
		Parameters
		----------
		M_grid : array-like
			grid of absolute magnitude at 1450A on which to evaluate the luminosity function
		z_bins : array-like
			redshift bins defined by their right edges. Default: None
		alphas : array-like, same shape as `z_bins`
			fits to alpha (bright-end slope of the double power-law luminosity function) corresponding element-wise to the `z_bins`. Default: None
		betas : array-like, same shape as `z_bins`
			fits to beta (faint-end slope of the double power-law luminosity function) corresponding element-wise to the `z_bins`. Default: None
		M_stars : array-like, same shape as `z_bins`
			fits to M_star (break magnitude) corresponding element-wise to the `z_bins`. Default: None
		fit_data : str
			sample on which alphas, betas, and M_stars were fit (one of ['combined']). Default: None

		"""
		if fit_data is None and (z_bins is None or alphas is None or betas is None or M_stars is None):
			raise ValueError("Either all the fit parameters or fit_data must be specified.")
		if not (fit_data is None or alphas is None or betas is None or M_stars is None):
			raise ValueError("Cannot specify fit parameters when fit_data is specified.")

		self.M_grid = M_grid
		self.z_bins = z_bins
		self.alphas = alphas
		self.betas = betas
		self.M_stars = M_stars

		if fit_data == 'combined':
			self._define_combined_fit_params()
		else:
			raise NotImplementedError

		n_bins = len(self.z_bins)
		if len(self.alphas) != n_bins:
			raise ValueError("z_bins and alphas should have the same length.")
		if len(self.betas) != n_bins:
			raise ValueError("z_bins and betas should have the same length.")
		if len(self.M_stars) != n_bins:
			raise ValueError("z_bins and M_stars should have the same length.")


	def _define_combined_fit_params(self):
		r"""Set the parameters fit on the combined sample of more than >80,000 color-selected AGN from ~14 datasets

		Note
		----
		The joint fit was done by [1]_ on the double power-law quasar luminosity function, e.g. [2]_. Note that the normalization (:math:`\phi^*`) is ignored because the luminosity function evaluated at the redshift bins is only used as a PMF from which to sample the luminosities, i.e. it's normalized to unity anyway.

		References
		----------
		.. [1] Kulkarni, Girish, Gábor Worseck, and Joseph F. Hennawi. "Evolution of the AGN UV luminosity function from redshift 7.5." Monthly Notices of the Royal Astronomical Society 488.1 (2019): 1035-1065.

		.. [2] Boyle, Brian John, et al. "The 2dF QSO Redshift Survey—I. The optical luminosity function of quasi-stellar objects." Monthly Notices of the Royal Astronomical Society 317.4 (2000): 1014-1022.

		"""
		self.z_bins = np.array([0.40, 0.60, 0.80, 1.00, 1.20,
		                       1.40, 1.60, 1.80, 2.20, 2.40, 
		                       2.50, 2.60, 2.70, 2.80, 2.90,
		                       3.00, 3.10, 3.20, 3.30, 3.40,
		                       3.50, 4.10, 4.70, 5.50, np.inf])
		self.alphas = -np.array([2.74, 3.49, 3.55, 3.69, 4.24,
		                        4.02, 4.35, 3.94, 4.26, 3.34,
		                        3.61, 3.31, 3.13, 3.78, 3.61, 
		                        5.01, 4.72, 4.39, 4.39, 4.76, 
		                        3.72, 4.84, 4.19, 4.55, 5.00])
		self.betas = -np.array([1.07, 1.55, 1.89, 1.88, 1.84, 
		                       1.88, 1.87, 1.69, 1.98, 1.61, 
		                       1.60, 1.38, 1.05, 1.34, 1.46, 
		                       1.71, 1.70, 1.96, 1.93, 2.08, 
		                       1.25, 2.07, 2.20, 2.31, 2.40])
		self.M_stars = -np.array([21.30, 23.38, 24.21, 24.60, 25.24,
		                         25.41, 25.77, 25.56, 26.35, 25.50,
		                         25.86, 25.33, 25.16, 25.94, 26.22,
		                         26.52, 26.48, 27.10, 27.19, 27.39,
		                         26.65, 27.26, 27.37, 27.89, 29.19])

	def get_double_power_law(self, alpha, beta, M_star):
		"""Evaluate the double power law at the given grid of absolute magnitudes

		Parameters
		----------
		alpha : float
			bright-end slope of the double power-law luminosity function
		beta : float
			faint-end slope of the double power-law luminosity function
		M_star : float
			break magnitude

		Note
		----
		Returned luminosity function is normalized to unity. See Note under `slope of the double power-law luminosity function`.

		Returns
		-------
		array-like
			the luminosity function evaluated at `self.M_grid` and normalized to unity

		"""
		denom = 10.0**(0.4*(alpha + 1.0)*(self.M_grid - M_star))
		denom += 10.0**(0.4*(beta + 1.0)*(self.M_grid - M_star))
		dn = 1.0/denom
		dn /= np.sum(dn)
		return dn

	def sample_agn_luminosity(self, z):
		"""Sample the AGN luminosity from the redshift-binned luminosity function

		Parameters
		----------
		z : float
			the AGN redshift

		Returns
		-------
		float
			sampled AGN luminosity at 1450A in mag
		
		"""
		# Assign redshift bin
		is_less_than_right_edge = (z < self.z_bins)
		alpha = self.alphas[is_less_than_right_edge][0]
		beta = self.betas[is_less_than_right_edge][0]
		M_star = self.M_stars[is_less_than_right_edge][0]

		# Evaluate function
		pmf = self.get_double_power_law(alpha, beta, M_star)

		# Sample luminosity
		sampled_M = np.random.choice(self.M_grid, None, replace=True, p=pmf)
		return sampled_M