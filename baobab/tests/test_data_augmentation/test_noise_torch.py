import random
import unittest
import numpy as np
import torch
from baobab.data_augmentation.noise_torch import NoiseModelTorch
from baobab.data_augmentation import get_noise_sigma2_lenstronomy

class TestNoiseTorch(unittest.TestCase):

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
                                num_exposures=1,
                                psf_type='GAUSSIAN',
                                kernel_point_source=None,
                                truncation=5,
                                #data_count_unit='ADU',
                                background_noise=None
                                )

    def test_get_noise_sigma2_lenstronomy_composite_background(self):
        """Validate the output of the lenstronomy noise module with background noise defined by sky and readout noise properties rather than an estimate of the combined background noise level

        """
        noise_sigma2 = get_noise_sigma2_lenstronomy(self.img, data_count_unit='ADU', **self.noise_kwargs)
        self.assertEqual(noise_sigma2['poisson'].shape, self.img.shape)
        self.assertTrue(isinstance(noise_sigma2['sky'], float))
        self.assertTrue(isinstance(noise_sigma2['readout'], float))
        return noise_sigma2

    def test_lenstronomy_vs_torch_ADU(self):
        """Compare the lenstronomy and torch noise variance for ADU units

        """
        numpy_sigma2 = get_noise_sigma2_lenstronomy(self.img, data_count_unit='ADU', **self.noise_kwargs)
        img_torch_tensor = torch.DoubleTensor(self.img)
        noise_model_torch = NoiseModelTorch(**self.noise_kwargs)
        torch_sigma2 = {}
        torch_sigma2['sky'] = noise_model_torch.get_sky_noise_sigma2()
        torch_sigma2['readout'] = noise_model_torch.get_readout_noise_sigma2()
        torch_sigma2['poisson'] = noise_model_torch.get_poisson_noise_sigma2(img_torch_tensor)
        self.assertEqual(numpy_sigma2['sky'], torch_sigma2['sky'])
        self.assertEqual(numpy_sigma2['readout'], torch_sigma2['readout'])
        np.testing.assert_array_almost_equal(numpy_sigma2['poisson'], torch_sigma2['poisson'].numpy(), decimal=7)

    def test_lenstronomy_vs_torch_electron(self):
        """Compare the lenstronomy and torch noise variance for electron units

        """
        numpy_sigma2 = get_noise_sigma2_lenstronomy(self.img, data_count_unit='e-', **self.noise_kwargs)
        img_torch_tensor = torch.DoubleTensor(self.img)
        noise_model_torch = NoiseModelTorch(data_count_unit='e-', **self.noise_kwargs)
        torch_sigma2 = {}
        torch_sigma2['sky'] = noise_model_torch.get_sky_noise_sigma2()
        torch_sigma2['readout'] = noise_model_torch.get_readout_noise_sigma2()
        torch_sigma2['poisson'] = noise_model_torch.get_poisson_noise_sigma2(img_torch_tensor)
        self.assertEqual(numpy_sigma2['sky'], torch_sigma2['sky'])
        self.assertEqual(numpy_sigma2['readout'], torch_sigma2['readout'])
        np.testing.assert_array_almost_equal(numpy_sigma2['poisson'], torch_sigma2['poisson'].numpy(), decimal=7)

if __name__ == '__main__':
    unittest.main()