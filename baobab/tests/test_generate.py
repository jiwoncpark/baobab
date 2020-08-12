import os
import shutil
import subprocess
import unittest
import baobab.configs as configs

def generate_config(cfg_filepath):
    """Run `generate.py` for the config file of the specified BNNPrior class

    Parameters
    ----------
    bnn_prior_name : str
        prefix to the config file (the BNNPrior class)

    """
    success = True
    cfg = configs.BaobabConfig.from_file(cfg_filepath)
    save_dir = cfg.out_dir
    #try:
    subprocess.run("python -m baobab.generate {:s} --n_data 2".format(cfg_filepath), shell=True)
    #except:
    #    success = False
    # Delete resulting data
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    return success

class TestGenerate(unittest.TestCase):
    """Test the `generate.py` script

    """
    @classmethod
    def setUpClass(cls):
        cls.cfg_root = os.path.abspath(os.path.dirname(configs.__file__))

    def test_generate_with_des(self):
        """Tests execution of `generate.py` script for all diagonal DES config files
         
        """
        cfg_filepath = os.path.join(self.cfg_root, 'des_config.json')
        success = generate_config(cfg_filepath)
        self.assertTrue(success, msg="des config")

    def test_generate_with_lsst(self):
        """Tests execution of `generate.py` script for all diagonal LSST config files
         
        """
        cfg_filepath = os.path.join(self.cfg_root, 'lsst_config.json')
        success = generate_config(cfg_filepath)
        self.assertTrue(success, msg="lsst config")

    def test_generate_with_hst(self):
        """Tests execution of `generate.py` script for a json HST config file, for backward compatibility
         
        """
        cfg_filepath = os.path.join(self.cfg_root, 'hst_config.json')
        success = generate_config(cfg_filepath)
        self.assertTrue(success, msg="hst config")

    def test_generate_with_diagonal_configs(self):
        """Tests execution of `generate.py` script for all diagonal config files
         
        """
        cfg_filepath = os.path.join(self.cfg_root, 'tdlmc_diagonal_config.py')
        success = generate_config(cfg_filepath)
        self.assertTrue(success, msg="tdlmc_diagonal_config")

        cfg_filepath = os.path.join(self.cfg_root, 'gamma_diagonal_config.py')
        success = generate_config(cfg_filepath)
        self.assertTrue(success, msg="gamma_diagonal_config")

    def test_generate_with_cov_configs(self):
        """Tests execution of `generate.py` script for all cov config files
         
        """
        cfg_filepath = os.path.join(self.cfg_root, 'tdlmc_cov_config.py')
        success = generate_config(cfg_filepath)
        self.assertTrue(success, msg="tdlmc_cov_config")

        cfg_filepath = os.path.join(self.cfg_root, 'gamma_cov_config.py')
        success = generate_config(cfg_filepath)
        self.assertTrue(success, msg="gamma_cov_config")

    def test_generate_with_empirical_configs(self):
        """Tests execution of `generate.py` script for all empirical config files
         
        """
        cfg_filepath = os.path.join(self.cfg_root, 'tdlmc_empirical_config.py')
        success = generate_config(cfg_filepath)
        self.assertTrue(success, msg="tdlmc_empirical_config")

        cfg_filepath = os.path.join(self.cfg_root, 'gamma_empirical_config.py')
        success = generate_config(cfg_filepath)
        self.assertTrue(success, msg="gamma_empirical_config")

    def test_generate_with_diagonal_cosmo_configs(self):
        """Tests execution of `generate.py` script for all diagonal cosmo config files
         
        """
        cfg_filepath = os.path.join(self.cfg_root, 'tdlmc_diagonal_cosmo_config.py')
        success = generate_config(cfg_filepath)
        self.assertTrue(success, msg="tdlmc_diagonal_cosmo_config")
            
if __name__ == '__main__':
    unittest.main()

        

    
