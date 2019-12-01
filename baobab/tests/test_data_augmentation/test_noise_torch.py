import os
import random
import unittest
import numpy as np
import torch
from baobab.data_augmentation import get_noise_sigma2_lenstronomy, NoiseModelTorch

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

    def test_build_tf_dataset(self):
        """Test whether tf.data.Dataset can be instantiated from tf.data.TFRecordDataset

        """
        tf_record_path = os.path.abspath('test_ADU')
        batch_size = 2
        n_epochs = 3
        generate_simple_tf_record(tf_record_path, tf_y_names)
        tf_dataset = tf.data.TFRecordDataset(tf_record_path).map(parse_example).repeat(n_epochs).shuffle(buffer_size=tf_data_size + 1).batch(batch_size, drop_remainder=True)
        
        images = [img for img, label in tf_dataset]
        labels = [label for img, label in tf_dataset]
        size = len(labels)
        np.testing.assert_array_equal(images[0].shape, (batch_size, tf_img_size, tf_img_size, 1))
        np.testing.assert_array_equal(labels[0].shape, (batch_size, len(tf_y_names)))
        np.testing.assert_equal(size, (tf_data_size*n_epochs//2))
        # Delete resulting data
        if os.path.exists(tf_record_path):
            os.remove(tf_record_path)

if __name__ == '__main__':
    unittest.main()