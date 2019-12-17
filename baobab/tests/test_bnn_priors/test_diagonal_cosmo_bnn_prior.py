import unittest

class TestDiagonalCosmoBNNPrior(unittest.TestCase):
    """A suite of tests alerting us for breakge, e.g. errors in
    instantiation of classes or execution of scripts, for DiagonalBNNPrior

    """
    def test_tdlmc_diagonal_cosmo_config(self):
        """Tests instantiation of TDLMC diagonal Config

        """
        import baobab.configs as configs
        cfg = configs.BaobabConfig.from_file(configs.tdlmc_diagonal_cosmo_config.__file__)
        return cfg

    def test_diagonal_cosmo_bnn_prior(self):
        """Tests instantiation and sampling of DiagonalBNNPrior

        """
        from baobab.bnn_priors import DiagonalCosmoBNNPrior
        cfg = self.test_tdlmc_diagonal_cosmo_config()
        diagonal_cosmo_bnn_prior = DiagonalCosmoBNNPrior(cfg.bnn_omega, cfg.components)
        return diagonal_cosmo_bnn_prior.sample()

if __name__ == '__main__':
    unittest.main()
