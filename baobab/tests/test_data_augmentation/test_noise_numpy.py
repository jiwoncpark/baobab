import random
import unittest
import numpy as np
import numpy
from baobab.data_augmentation import get_noise_sigma2_lenstronomy, NoiseModelNumpy

class TestNoiseNumpy(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Seed randomness

        """
        np.random.seed(123)
        random.seed(123)
        cls.img = np.random.randn(3, 3)*3.0 + 3.0
        cls.noise_kwargs = dict(
                                pixel_scale=0.08,
                                exposure_time=100.0,
                                magnitude_zero_point=25.9463,
                                read_noise=10,
                                ccd_gain=7.0,
                                sky_brightness=20.1,
                                seeing=0.6,
                                num_exposures=2,
                                psf_type='GAUSSIAN',
                                kernel_point_source=None,
                                truncation=5,
                                #data_count_unit='ADU',
                                background_noise=None
                                )

    def test_lenstronomy_background(self):
        """Without involving the numpy noise module, simply check that the background noise variance (readout plus sky) when background noise is specified as None equals the computed readout plus sky variance"""
        lens_sigma2 = get_noise_sigma2_lenstronomy(self.img, data_count_unit='ADU', **self.noise_kwargs)
        if self.noise_kwargs['background_noise'] is None:
            lens_sigma2['sky_plus_readout'] = lens_sigma2['sky'] + lens_sigma2['readout']

    def test_get_noise_sigma2_lenstronomy_composite_background(self):
        """Validate the output of the lenstronomy noise module with background noise defined by sky and readout noise properties rather than an estimate of the combined background noise level

        """
        lens_sigma2 = get_noise_sigma2_lenstronomy(self.img, data_count_unit='ADU', **self.noise_kwargs)
        self.assertEqual(lens_sigma2['poisson'].shape, self.img.shape)
        self.assertTrue(isinstance(lens_sigma2['sky'], float))
        self.assertTrue(isinstance(lens_sigma2['readout'], float))
        return lens_sigma2

    def test_lenstronomy_vs_numpy_ADU(self):
        """Compare the lenstronomy and numpy noise variance for ADU units

        """
        lens_sigma2 = get_noise_sigma2_lenstronomy(self.img, data_count_unit='ADU', **self.noise_kwargs)
        noise_model_numpy = NoiseModelNumpy(**self.noise_kwargs)
        numpy_sigma2 = {}
        numpy_sigma2['sky'] = noise_model_numpy.get_sky_noise_sigma2()
        numpy_sigma2['readout'] = noise_model_numpy.get_readout_noise_sigma2()
        numpy_sigma2['poisson'] = noise_model_numpy.get_poisson_noise_sigma2(self.img)
        np.testing.assert_almost_equal(lens_sigma2['sky'], numpy_sigma2['sky'], decimal=7, err_msg="sky")
        np.testing.assert_almost_equal(lens_sigma2['readout'], numpy_sigma2['readout'], decimal=7, err_msg="readout")
        np.testing.assert_array_almost_equal(lens_sigma2['poisson'], numpy_sigma2['poisson'], decimal=7, err_msg="poisson")

    def test_lenstronomy_vs_numpy_electron(self):
        """Compare the lenstronomy and numpy noise variance for electron units

        """
        lens_sigma2 = get_noise_sigma2_lenstronomy(self.img, data_count_unit='e-', **self.noise_kwargs)
        noise_model_numpy = NoiseModelNumpy(data_count_unit='e-', **self.noise_kwargs)
        numpy_sigma2 = {}
        numpy_sigma2['sky'] = noise_model_numpy.get_sky_noise_sigma2()
        numpy_sigma2['readout'] = noise_model_numpy.get_readout_noise_sigma2()
        numpy_sigma2['poisson'] = noise_model_numpy.get_poisson_noise_sigma2(self.img)
        np.testing.assert_almost_equal(lens_sigma2['sky'], numpy_sigma2['sky'], decimal=7, err_msg="sky")
        np.testing.assert_almost_equal(lens_sigma2['readout'], numpy_sigma2['readout'], decimal=7, err_msg="readout")
        np.testing.assert_array_almost_equal(lens_sigma2['poisson'], numpy_sigma2['poisson'], decimal=7, err_msg="poisson")

if __name__ == '__main__':
    unittest.main()