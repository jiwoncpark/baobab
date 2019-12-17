import random
import numpy as np
import unittest
from scipy.stats import kurtosis, skew
from scipy.special import gamma

class TestDistributions(unittest.TestCase):
    """A suite of tests verifying that the input PDFs and the sample distributions
    match.

    """

    @classmethod
    def setUpClass(cls):
        """Instantiate DiagonalBNNPrior and Config as class objects

        """
        from baobab.bnn_priors import DiagonalBNNPrior
        import baobab.configs as configs
        cfg = configs.BaobabConfig.from_file(configs.tdlmc_diagonal_config.__file__)
        cls.cfg = cfg
        cls.diagonal_bnn_prior = DiagonalBNNPrior(cfg.bnn_omega, cfg.components)
        np.random.seed(123)
        random.seed(123)

    def test_generalized_normal(self):
        """Test the generalized normal sampling

        """
        from baobab.distributions import sample_generalized_normal
        mu = 0.0
        alpha = 0.5
        p = 10.0
        n_samples = 10**4
        sample = np.zeros((n_samples,))
        for i in range(n_samples):
            sample[i] = sample_generalized_normal(mu=mu, alpha=alpha, p=p)
        sample_mean = np.mean(sample)
        sample_var = np.var(sample)
        sample_skew = skew(sample)
        sample_kurtosis = kurtosis(sample)
        #sample_entropy = entropy(sample)
        exp_mean = mu
        exp_var = alpha**2.0 * gamma(3/p) / gamma(1/p)
        exp_skew = 0
        exp_kurtosis = gamma(5/p) * gamma(1/p) / gamma(3/p)**2.0 - 3.0
        #exp_entropy = 1/p - np.log(p / (2 * alpha * gamma(1/p)))
        precision = 2
        np.testing.assert_almost_equal(sample_mean, exp_mean, precision)
        np.testing.assert_almost_equal(sample_var, exp_var, precision)
        np.testing.assert_almost_equal(sample_skew, exp_skew, precision)
        np.testing.assert_almost_equal(sample_kurtosis, exp_kurtosis, precision)
        #np.testing.assert_almost_equal(sample_entropy, exp_entropy, precision)

if __name__ == '__main__':
    unittest.main()

