import os, sys
import shutil
import subprocess
import unittest
import numpy as np

class TestEmpiricalBNNPrior(unittest.TestCase):
    """A suite of tests alerting us for breakge, e.g. errors in
    instantiation of classes or execution of scripts, for EmpiricalBNNPrior

    """
    def test_tdlmc_empirical_config(self):
        """Tests instantiation of TDLMC diagonal Config

        """
        import baobab.configs as configs
        cfg = configs.BaobabConfig.from_file(configs.tdlmc_empirical_config.__file__)
        return cfg

    def test_empirical_bnn_prior(self):
        """Tests instantiation and sampling of EmpiricalBNNPrior

        """
        from baobab.bnn_priors import EmpiricalBNNPrior
        cfg = self.test_tdlmc_empirical_config()
        empirical_bnn_prior = EmpiricalBNNPrior(cfg.bnn_omega, cfg.components)
        return empirical_bnn_prior.sample()

if __name__ == '__main__':
    unittest.main()
