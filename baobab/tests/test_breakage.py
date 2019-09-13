import os, sys
import shutil
from unittest import TestCase

class TestBreakage(TestCase):
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
        from 
        """
        cfg = self.test_tdlmc_diagonal_config()
        for cfg_filename in dir('baobab/configs'):
            if cfg_filename.endswith('_config'):
                try:
                    # tdlmc_diagonal_config.py
                    os.system('generate baobab/configs/{:s} 2'.format(cfg_filename))
                except RuntimeError:
                    print("generate.py script is broken.")
                # Delete resulting data
                if os.path.exists(cfg.out_dir):
                    shutil.rmtree(cfg.out_dir)
            


        

    
