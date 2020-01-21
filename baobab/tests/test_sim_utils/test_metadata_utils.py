import unittest
import numpy as np
import pandas as pd
import baobab.sim_utils.metadata_utils as metadata_utils

class TestMetadataUtils(unittest.TestCase):
    """Tests for the metadata utils module used to convert between parameter definitions

    """
    def test_g1g2_vs_gamma_psi_symmetry(self):
        n_data = 1000
        data = {
                'external_shear_gamma_ext': np.abs(np.random.randn(n_data)*0.01 + 0.025),
                'external_shear_psi_ext': np.random.rand(n_data)*np.pi - np.pi*0.5,
                }
        only_gamma_psi = pd.DataFrame(data)
        g1g2_added = metadata_utils.add_g1g2_columns(only_gamma_psi)
        del only_gamma_psi
        only_g1g2 = g1g2_added[['external_shear_gamma1', 'external_shear_gamma2']].copy()
        gamma_psi_added = metadata_utils.add_gamma_psi_ext_columns(only_g1g2)
        del only_g1g2
        np.testing.assert_array_almost_equal(g1g2_added['external_shear_gamma1'].values, gamma_psi_added['external_shear_gamma1'].values, err_msg="gamma1")
        np.testing.assert_array_almost_equal(g1g2_added['external_shear_gamma2'].values, gamma_psi_added['external_shear_gamma2'].values, err_msg="gamma2")
        np.testing.assert_array_almost_equal(g1g2_added['external_shear_gamma_ext'].values, gamma_psi_added['external_shear_gamma_ext'].values, err_msg="gamma_ext")
        np.testing.assert_array_almost_equal(g1g2_added['external_shear_psi_ext'].values, gamma_psi_added['external_shear_psi_ext'].values, err_msg="psi_ext")

    def test_e1e2_vs_qphi_symmetry(self):
        n_data = 1000
        data = {
                'a_e1': np.random.randn(n_data)*0.1,
                'a_e2': np.random.randn(n_data)*0.1,
                }
        only_e1e2 = pd.DataFrame(data)
        pass

if __name__ == '__main__':
    unittest.main()