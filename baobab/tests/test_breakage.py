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

    def test_generate(self):
        """Tests execution of `generate.py` script
        from 
        """
        cfg = self.test_tdlmc_diagonal_config()
        try:
            os.system('generate baobab/configs/tdlmc_diagonal_config.py 2')
        except RuntimeError:
            print("generate.py script is broken.")
        # Delete resulting data
        if os.path.exists(cfg.out_dir):
            shutil.rmtree(cfg.out_dir)
            


        

    
