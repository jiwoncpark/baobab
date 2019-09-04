from unittest import TestCase

class TestBreakage(TestCase):
    """A suite of tests alerting us for breakge, e.g. errors in
    instantiation of classes or execution of scripts

    """
    def test_config(self):
        """Tests instantiation of Config

        """
        from baobab import configs
        cfg = configs.Config.fromfile(configs.tdlmc_config.__file__)
        return cfg

    def test_diagonal_bnn_prior(self):
        """Tests instantiation of DiagonalBNNPrior

        """
        from baobab.bnn_priors import DiagonalBNNPrior
        cfg = self.test_config()
        diagonal_bnn_prior = DiagonalBNNPrior(cfg.bnn_omega, cfg.components)

    def test_generate(self):
        """Tests execution of `generate.py` script

        """
        pass
    
