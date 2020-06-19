import unittest
import numpy as np
from baobab.sim_utils import instantiate_PSF_kwargs, instantiate_PSF_models, get_PSF_model

class TestPSFUtils(unittest.TestCase):
    """Tests for the module used to create PSFs.

    """
    def test_instantiate_PSF_kwargs(self):
        """Test whether the correct kwargs are generated

        """
        psf_dict = dict(
        	type='PIXEL',
        	kernel_size=91,
        	which_psf_maps=None
        	)
        pixel_scale = 0.08
        psf_kwargs = instantiate_PSF_kwargs(psf_dict,pixel_scale)

        # Test that the correct number of PSFs are generated
        self.assertEqual(len(psf_kwargs),16)

        # Test that the psf kwargs have the expected values
        for kwargs in psf_kwargs:
        	self.assertEqual(kwargs['psf_type'],'PIXEL')
        	self.assertEqual(kwargs['pixel_size'],0.08)
        	self.assertEqual(kwargs['kernel_point_source'].shape,(91,91))

        # Pick a specific psf map
        psf_dict = dict(
        	type='PIXEL',
        	kernel_size=91,
        	which_psf_maps=101
        	)
        psf_kwargs = instantiate_PSF_kwargs(psf_dict,pixel_scale)

        # Test that the correct number of PSFs are generated
        self.assertEqual(len(psf_kwargs),1)

        # Test that the psf kwargs have the expected values
        for kwargs in psf_kwargs:
        	self.assertEqual(kwargs['psf_type'],'PIXEL')
        	self.assertEqual(kwargs['pixel_size'],0.08)
        	self.assertEqual(kwargs['kernel_point_source'].shape,(91,91))

        # Pick a Gaussian psf
        psf_dict = dict(
        	type='GAUSSIAN',
        	kernel_size=91,
        	PSF_FWHM=0.1
        	)
        psf_kwargs = instantiate_PSF_kwargs(psf_dict,pixel_scale)

        # Test that the correct number of PSFs are generated
        self.assertEqual(len(psf_kwargs),1)

        # Test that the psf kwargs have the expected values
        for kwargs in psf_kwargs:
        	self.assertEqual(kwargs['psf_type'],'GAUSSIAN')
        	self.assertEqual(kwargs['fwhm'],0.1)
        	self.assertEqual(kwargs['pixel_size'],0.08)

    def test_instantiate_PSF_models(self):
        """Test whether the kwargs give the correct psfs.

        """
        psf_dict = dict(
        	type='PIXEL',
        	kernel_size=91,
        	which_psf_maps=None
        	)
        pixel_scale = 0.08
        psf_models = instantiate_PSF_models(psf_dict,pixel_scale)

        # Test that the correct number of PSFs are generated
        self.assertEqual(len(psf_models),16)

        # Pick a specific psf map
        psf_dict = dict(
        	type='PIXEL',
        	kernel_size=91,
        	which_psf_maps=101
        	)
        psf_models = instantiate_PSF_models(psf_dict,pixel_scale)

        # Test that the correct number of PSFs are generated
        self.assertEqual(len(psf_models),1)

        # Pick a Gaussian psf
        psf_dict = dict(
        	type='GAUSSIAN',
        	kernel_size=91,
        	PSF_FWHM=0.1
        	)
        psf_models = instantiate_PSF_models(psf_dict,pixel_scale)

        # Test that the correct number of PSFs are generated
        self.assertEqual(len(psf_models),1)

if __name__ == '__main__':
    unittest.main()