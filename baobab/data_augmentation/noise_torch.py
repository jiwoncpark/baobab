import torch
import lenstronomy.Util.data_util as data_util
__all__ = ['NoiseModelTorch']

class NoiseModelTorch:
    """A combination of sky, readout, and Poisson flux noise to be added to the image

    Note
    ----
    This is a torch wrapper around the functionality provided by the `SingleBand` class in lenstronomy.

    """
    def __init__(self, pixel_scale, exposure_time, magnitude_zero_point, read_noise=None, ccd_gain=None, sky_brightness=None, seeing=None, num_exposures=1, psf_type='GAUSSIAN', kernel_point_source=None, truncation=5, data_count_unit='ADU', background_noise=None):
        """

        Parameters
        ----------
        pixel_scale : float
            pixel scale in arcsec/pixel
        exposure_time : float
            exposure time per image in seconds
        magnitude_zero_point : float
            magnitude at which 1 count per second per arcsecond square is registered
        read_noise : float
            std of noise generated by readout (in units of electrons)
        ccd_gain : float
            electrons/ADU (analog-to-digital unit). A gain of 8 means that the camera digitizes the CCD signal so that each ADU corresponds to 8 photoelectrons
        sky_brightness : float
             sky brightness (in magnitude per square arcsec)
        seeing : float
            fwhm of PSF
        num_exposures : float
            number of exposures that are combined
        psf_type : str
            type of PSF ('GAUSSIAN' and 'PIXEL' supported)
        kernel_point_source : 2d numpy array
            model of PSF centered with odd number of pixels per axis(optional when psf_type='PIXEL' is chosen)
        truncation : float
            Gaussian truncation (in units of sigma), only required for 'GAUSSIAN' model
        data_count_unit : str
            unit of the data (and other properties), 'e-': (electrons assumed to be IID), 'ADU': (analog-to-digital unit)
        background_noise : float
            sqrt(variance of background) as a total contribution from read noise, sky brightness, etc. in units of the data_count_units
            If you set this parameter, it will override readout_noise, sky_brightness. Default: None

        """
        self.pixel_scale = pixel_scale
        self.exposure_time = exposure_time
        self.magnitude_zero_point = magnitude_zero_point
        self.ccd_gain = ccd_gain
        self.sky_brightness = sky_brightness
        self.seeing = seeing
        self.num_exposures = num_exposures
        self.psf_type = psf_type
        self.kernel_point_source = kernel_point_source
        self.truncation = truncation
        self.data_count_unit = data_count_unit
        self.background_noise = background_noise

        #FIXME: seeing, psf_type, kernel_point_source, and truncation do not seem to be used at all.
        
        self.readout_noise = read_noise
        if self.data_count_unit == 'ADU':
            self.readout_noise /= self.ccd_gain

        self.sky_brightness = data_util.magnitude2cps(self.sky_brightness, self.magnitude_zero_point)
        if self.data_count_unit == 'e-':
            self.sky_brightness *= self.ccd_gain

        self.exposure_time_tot = self.num_exposures * self.exposure_time
        self.readout_noise_tot = self.num_exposures * self.readout_noise**2.0
        self.sky_per_pixel = self.sky_brightness * pixel_scale**2.0

        self.get_background_noise_sigma2 = getattr(self, 'get_background_noise_sigma2_composite') if self.background_noise is None else getattr(self, 'get_background_noise_sigma2_simple')
        
        self.scaled_exposure_time = self.exposure_time
        if self.data_count_unit == 'ADU':
            self.scaled_exposure_time *= self.ccd_gain

    def get_sky_noise_sigma2(self):
        """Compute the variance in sky noise

        Returns
        -------
        float
            variance of the sky noise, in cps^2

        """
        return self.sky_per_pixel**2.0 / self.exposure_time_tot

    def get_readout_noise_sigma2(self):
        """Compute the variance in readout noise

        Returns
        -------
        float
            variance of the readout noise, in cps^2

        """
        return self.readout_noise_tot / self.exposure_time_tot**2.0

    def get_background_noise_sigma2_simple(self):
        """Get the variance in background noise from the specified estimate of the background noise, rather than computing it from the sky brightness and read noise

        Returns
        -------
        float
            variance of the background noise, in cps^2

        """
        return self.background**2.0

    def get_background_noise_sigma2_composite(self):
        """Get the variance in background noise from the sky brightness and read noise

        Returns
        -------
        float
            variance of the background noise, in cps^2

        """
        return self.get_sky_noise_sigma2() + self.get_readout_noise_sigma2()

    def get_poisson_noise_sigma2(self, img):
        """Get the variance in Poisson flux noise from the image

        Parameters
        ----------
        img : 2D torch.Tensor 
            the image of flux values in cps on which to evaluate the noise

        Returns
        -------
        float
            variance of the Poisson flux noise, in cps^2

        """
        return torch.max(img, torch.zeros_like(img))/self.scaled_exposure_time

    def get_noise_sigma2(self, img):
        """Get the variance of total noise due to the combined effects of sky, readout, and Poisson flux noise

        Parameters
        ----------
        img : 2D torch.Tensor 
            the image of flux values in cps on which to evaluate the noise

        Returns
        -------
        2D torch.Tensor 
            variance of total noise, in cps^2

        """
        return self.get_background_noise_sigma2() + self.get_poisson_noise_sigma2(img)

    def get_noise_map(self, img):
        """Get the total random noise map due to the combined effects of sky, readout, and Poisson flux noise

        Parameters
        ----------
        img : 2D torch.Tensor 
            the image of flux values in cps on which to evaluate the noise

        Returns
        -------
        2D torch.Tensor 
            the noise map in cps

        """
        return torch.randn_like(img)*self.get_noise_sigma2(img)**0.5