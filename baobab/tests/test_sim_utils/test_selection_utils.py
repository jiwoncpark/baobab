import unittest
import numpy as np
from baobab.sim_utils import Selection

class TestSelectionUtils(unittest.TestCase):
    """Tests for the selection module used to accept or reject sampled parameters

    """
    def test_e1e2_rejection(self):
        """Test whether a sample is rejected based on the ellipticity selection criterion when it should be

        """
        selection_cfg = {'initial': []}
        selection = Selection(selection_cfg, ['lens_mass', 'external_shear', 'src_light'])
        sample = {'lens_mass': {'e1': 0.9, 'e2': 0.9},
        'src_light': {'e1': 0.01, 'e2': 0.01}}
        np.testing.assert_equal(selection.reject_initial(sample), True)

    def test_theta_E_rejection(self):
        """Test whether a sample is rejected based on the Einstein radius selection criterion when it should be

        """
        selection_cfg = {'initial': ["lambda x: x['lens_mass']['theta_E'] > 0.5",]}
        selection = Selection(selection_cfg, ['lens_mass', 'external_shear', 'src_light'])
        sample = {'lens_mass': {'e1': 0.01, 'e2': 0.01, 'theta_E': 0.1},
        'src_light': {'e1': 0.01, 'e2': 0.01}}
        np.testing.assert_equal(selection.reject_initial(sample), True)

if __name__ == '__main__':
    unittest.main()
