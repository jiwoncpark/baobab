import os, sys
import shutil
import subprocess
import unittest

class TestBreakage(unittest.TestCase):
    """A suite of tests alerting us for breakge, e.g. errors in
    instantiation of classes or execution of scripts

    """
    def test_tdlmc_diagonal_config(self):
        """Tests instantiation of TDLMC diagonal Config

        """
        from baobab import configs
        cfg = configs.Config.fromfile(configs.tdlmc_diagonal_config.__file__)
        return cfg

    def test_diagonal_bnn_prior(self):
        """Tests instantiation of DiagonalBNNPrior

        """
        from baobab.bnn_priors import DiagonalBNNPrior
        cfg = self.test_tdlmc_diagonal_config()
        diagonal_bnn_prior = DiagonalBNNPrior(cfg.bnn_omega, cfg.components)

    def test_tdlmc_cov_config(self):
        """Tests instantiation of TDLMC diagonal Config

        """
        from baobab import configs
        cfg = configs.Config.fromfile(configs.tdlmc_cov_config.__file__)
        return cfg

    def test_cov_bnn_prior(self):
        """Tests instantiation of CovBNNPrior

        """
        from baobab.bnn_priors import CovBNNPrior
        cfg = self.test_tdlmc_cov_config()
        cov_bnn_prior = CovBNNPrior(cfg.bnn_omega, cfg.components)

    def test_generate(self):
        """Tests execution of `generate.py` script for all template config files
         
        """

        from baobab import configs
        cfg_root = os.path.abspath(os.path.dirname(configs.__file__))
        n_failures = 0
        for cfg_filename in os.listdir(cfg_root):
            if cfg_filename.endswith('_config.py'):
                cfg_filepath = os.path.join(cfg_root, cfg_filename)
                cfg = configs.Config.fromfile(cfg_filepath)
                save_dir = os.path.join('out_data', cfg.out_dir)
                try:
                    subprocess.check_output('generate {:s} --n_data 2'.format(cfg_filepath), shell=True)
                except:
                    n_failures += 1
                # Delete resulting data
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
        self.assertTrue(n_failures==0) # FIXME: clumsy

    def test_to_hdf5(self):
        """Tests execution of `to_hdf5.py` script for all template config files
         
        """
        from baobab import configs
        cfg_filepath = configs.tdlmc_diagonal_config.__file__
        cfg = configs.Config.fromfile(cfg_filepath)
        subprocess.check_output('generate {:s} --n_data 5'.format(cfg_filepath), shell=True)
        save_dir = os.path.join('out_data', cfg.out_dir)
        n_failures = 0
        for channel_format in ['tf', 'theano']:
            try:
                subprocess.check_output('to_hdf5 {:s} --format {:s}'.format(save_dir, channel_format), shell=True)
            except:
                n_failures += 1
        # Delete resulting data
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        self.assertTrue(n_failures==0) # FIXME: clumsy
            
if __name__ == '__main__':
    unittest.main()

        

    
