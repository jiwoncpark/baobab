from unittest import TestCase

class TestInstantiation(TestCase):
    def instantiate_config(self):
        from baobab import configs
        cfg = configs.Cfg('../configs/test_config.py')
        return cfg

    def instantiate_diagonal_bnn_prior(self):
        from baobab.bnn_priors import DiagonalBNNPrior
        cfg = self.instantiate_config
        diagonal_bnn_prior = DiagonalBNNPrior(cfg.bnn_omega, cfg.components)
    
