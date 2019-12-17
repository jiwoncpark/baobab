import unittest

class TestDiagonalBNNPrior(unittest.TestCase):
    """A suite of tests alerting us for breakge, e.g. errors in
    instantiation of classes or execution of scripts, for DiagonalBNNPrior

    """
    def test_tdlmc_diagonal_config(self):
        """Tests instantiation of TDLMC diagonal Config

        """
        import baobab.configs as configs
        cfg = configs.BaobabConfig.from_file(configs.tdlmc_diagonal_config.__file__)
        return cfg

    def test_diagonal_bnn_prior(self):
        """Tests instantiation and sampling of DiagonalBNNPrior

        """
        from baobab.bnn_priors import DiagonalBNNPrior
        cfg = self.test_tdlmc_diagonal_config()
        diagonal_bnn_prior = DiagonalBNNPrior(cfg.bnn_omega, cfg.components)
        return diagonal_bnn_prior.sample()

if __name__ == '__main__':
    unittest.main()
