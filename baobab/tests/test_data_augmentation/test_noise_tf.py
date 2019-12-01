import os
import random
import unittest
import numpy as np
import tensorflow as tf
from baobab.data_augmentation import get_noise_sigma2_lenstronomy, NoiseModelTF
from baobab.tests.test_data_augmentation.tf_data_utils import generate_simple_tf_record, parse_example, tf_img_size, tf_y_names, tf_data_size

class TestNoiseTF(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Seed randomness

        """
        np.random.seed(123)
        random.seed(123)
        cls.img = np.random.randn(3, 3)*3.0 + 6.0
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

    def test_lenstronomy_vs_tf_ADU(self):
        """Compare the lenstronomy and tf noise variance for ADU units

        """
        numpy_sigma2 = get_noise_sigma2_lenstronomy(self.img, data_count_unit='ADU', **self.noise_kwargs)
        img_tf_tensor = tf.cast(self.img, tf.float32)
        noise_model_tf = NoiseModelTF(**self.noise_kwargs)
        tf_sigma2 = {}
        tf_sigma2['sky'] = noise_model_tf.get_sky_noise_sigma2()
        tf_sigma2['readout'] = noise_model_tf.get_readout_noise_sigma2()
        tf_sigma2['poisson'] = noise_model_tf.get_poisson_noise_sigma2(img_tf_tensor)
        np.testing.assert_array_almost_equal(self.img, img_tf_tensor.numpy(), decimal=5)
        np.testing.assert_equal(numpy_sigma2['sky'], tf_sigma2['sky'])
        np.testing.assert_equal(numpy_sigma2['readout'], tf_sigma2['readout'])
        np.testing.assert_array_almost_equal(numpy_sigma2['poisson'], tf_sigma2['poisson'].numpy(), decimal=7)

    def test_lenstronomy_vs_tf_electron(self):
        """Compare the lenstronomy and tf noise variance for electron units

        """
        numpy_sigma2 = get_noise_sigma2_lenstronomy(self.img, data_count_unit='e-', **self.noise_kwargs)
        img_tf_tensor = tf.cast(self.img, tf.float32)
        noise_model_tf = NoiseModelTF(data_count_unit='e-', **self.noise_kwargs)
        tf_sigma2 = {}
        tf_sigma2['sky'] = noise_model_tf.get_sky_noise_sigma2()
        tf_sigma2['readout'] = noise_model_tf.get_readout_noise_sigma2()
        tf_sigma2['poisson'] = noise_model_tf.get_poisson_noise_sigma2(img_tf_tensor)
        np.testing.assert_array_almost_equal(self.img, img_tf_tensor.numpy(), decimal=5)
        np.testing.assert_equal(numpy_sigma2['sky'], tf_sigma2['sky'])
        np.testing.assert_equal(numpy_sigma2['readout'], tf_sigma2['readout'])
        np.testing.assert_array_almost_equal(numpy_sigma2['poisson'], tf_sigma2['poisson'].numpy(), decimal=7)

    def test_build_tf_dataset(self):
        """Test whether tf.data.Dataset can be instantiated from tf.data.TFRecordDataset with the data augmentation (noise addition) mapping

        """
        tf_record_path = os.path.abspath('test_ADU')
        batch_size = 2
        n_epochs = 3

        noise_model_tf = NoiseModelTF(**self.noise_kwargs)
        add_noise_func = getattr(noise_model_tf, 'add_noise')
        #print(add_noise_func(tf.ones((3, 3), dtype=tf.float32)))

        generate_simple_tf_record(tf_record_path, tf_y_names)
        tf_dataset = tf.data.TFRecordDataset(tf_record_path).map(parse_example).map(lambda image, label: (add_noise_func(image), label)).repeat(n_epochs).shuffle(buffer_size=tf_data_size + 1).batch(batch_size, drop_remainder=True)
        
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