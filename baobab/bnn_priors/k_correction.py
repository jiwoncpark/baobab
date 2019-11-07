import numpy as np

class KCorrector:
    r"""Applies K correction, i.e. "corrects" for the redshift and bandpass throughput in the conversion between apparent and absolute magnitudes

    Note
    ----
    The following is summarized from [1]_. Consider a source observed to have apparent magnitude :math:`m_R` when observed through photometric bandpass :math:`R`, for which we want to know the absolute magnitude :math:`M_Q` in emitted-frame bandpass :math:`Q`. Denote the distance modulus as :math:`DM`. The K correction term :math:`K_{QR}` for this source is defined as

    .. math::

        m_R = M_Q + DM + K_{QR}

    We can compute :math:`K_{QR}` via

    .. math::

        K_{QR} = -2.5 \log_{10} \left[ \frac{1}{1 + z} \frac{\int d\lambda_0 \lambda_0 f_\lambda(\lambda_0) R(\lambda_0) \int d\lambda_e \lambda_e g_\lambda^Q(\lambda_e) Q(\lambda_e) }{ \int d\lambda_0 \lambda_0 g_\lambda^R(\lambda_0) R(\lambda_0) \int d\lambda_e f_\lambda\left((1 + z)\lambda_e \right) Q(\lambda_e)} \right]

    where :math:`f_\lambda` is the spectral density of flux and the superscript denotes 

    References
    ----------
    .. [1] Hogg, David W., et al. "The K correction." arXiv preprint astro-ph/0210394 (2002).

    """
    def __init__(self, z_src, sed_src, standard_sed_src, bandpass_src, standard_sed_obs, bandpass_obs):
        """
        Parameters
        ----------
        sed_src : `SED` object
            the SED of the source.
        standard_sed_src : 'SED' object
            the spectral density of flux for the zero-magnitude ("standard") source, with which the absolute magnitude is defined
        bandpass_src : `Bandpass` object
            the photometric bandpass throughput at the source's rest (emitted) frame, with which the absolute magnitude is defined
        standard_sed_obs : 'SED' object
            the spectral density of flux for the zero-magnitude ("standard") source, with which the apparent magnitude is defined
        bandpass_obs : 'Bandpass' object
            the photometric bandpass throughput at the observed frame, with which the apparent magnitude is defined

        Note
        ----
        The bandpass throughput is dimensionless and can be interpreted as the probability of detecting a photon with a given wavelength. (See docstring for `Bandpass` object). The dimensions of the supplied `sed` properties of `sed_src`, `standard_sed_src`, and `standard_sed_obs` need not be in physical units (e.g. W m^{-2} nm^{-1}) but must be the same.

        """
        self.sed_src = sed_src
        self.standard_sed_src = standard_sed_src
        self.bandpass_src = bandpass_src
        self.standard_sed_obs = standard_sed_obs
        self.bandpass_obs = bandpass_obs

    @property
    def sed_src(self):
        """Get the source SED

        """
        return self.sed_src

    @sed_src.setter
    def sed_src(self, new_sed_src):
        """Set a new source SED

        """
        self.sed_src = sed_src

    def get_k_correction(self, z_src):
        """Calculate the K correction factor

        Parameters
        ----------
        z_src : float
            the source redshift
        
        Returns
        -------
        float
            the K correction factor

        """
        pass

class ChromaticObject:
    """General class for chromatic objects, or objects carrying wavelength-dependent properties

    """
    def __init__(self, wavelengths, chromatic_property):
        """
        Parameters
        ----------
        wavelengths : array-like
            wavelengths in nm at which the `chromatic_property` is defined, sorted in increasing order
        chromatic_property : array-like
            the chromatic property, e.g. bandpass throughput or spectral density, evaluated elementwise at `wavelengths`
        
        """
        self.wavelengths = wavelengths
        self.chromatic_property = chromatic_property

    def __call__(self, eval_wavelengths):
        """Evaluates the chromatic property via linear interpolation
        
        Parameters
        ----------
        eval_wavelengths : array-like
            wavelengths at which to evaluate the chromatic property

        """
        interpolated = np.interp(eval_wavelengths, self.wavelengths, self.chromatic_property)
        return interpolated

class Bandpass(ChromaticObject):
    """Represents the instrument bandpass, characterized by its throughput

    """
    def __init__(self, wavelengths, throughput):
        """
        Parameters
        ----------
        wavelengths : array-like
            wavelengths in nm at which the `throughput` is defined
        throughput : array-like
            the bandpass throughput for a photon counter, i.e. the probability that a photon with a given wavelength gets counted
        template : str


        """
        super(Bandpass, self).__init__(wavelengths, throughput)
        self.throughput = self.throughput

    @classmethod
    def from_name(cls, bandpass_name):
        """Alternate constructor based on the name of bandpass for a select set of telescopes

        Parameters
        ----------
        bandpass_name : str
            the name of bandpass (one of ['LSST_<ugrizy>', 'WFC3_IR_F160W'])

        """

class SED(ChromaticObject):
    """Represents the spectral energy distribution (SED) of a source

    """
    def __init__(self, wavelengths, sed):
        """
        Parameters
        ----------
        wavelengths : array-like
            wavelengths in nm at which `sed` is defined
        sed : array-like
            spectral density in units of W m^{-2} s^{-1} nm^{-1}
        template : str

        """
        super(SED, self).__init__(wavelengths, sed)
        self.sed = sed

    @classmethod
    def from_name(cls, source_name):
        """Alternate constructor based on the name of the source for a select set of sources

        Parameters
        ----------
        source_name : str
            the name of source (one of ['agn_dc2'])

        """
        