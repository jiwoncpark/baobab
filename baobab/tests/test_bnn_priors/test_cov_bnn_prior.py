import unittest

class TestCovBNNPrior(unittest.TestCase):
    """A suite of tests alerting us for breakge, e.g. errors in
    instantiation of classes or execution of scripts, for CovBNNPrior

    """
    def test_tdlmc_cov_config(self):
        """Tests instantiation of TDLMC diagonal Config

        """
        import baobab.configs as configs
        cfg = configs.BaobabConfig.from_file(configs.tdlmc_cov_config.__file__)
        return cfg

    def test_cov_bnn_prior(self):
        """Tests instantiation and sampling of CovBNNPrior

        """
        from baobab.bnn_priors import CovBNNPrior
        cfg = self.test_tdlmc_cov_config()
        cov_bnn_prior = CovBNNPrior(cfg.bnn_omega, cfg.components)
        return cov_bnn_prior.sample()

if __name__ == '__main__':
    unittest.main()
