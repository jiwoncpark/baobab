import random
import numpy as np
import unittest
from scipy.stats import kurtosis, skew
from scipy.special import gamma
import baobab.distributions as bb_dist

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
        mu = 0.0
        alpha = 0.5
        p = 10.0
        n_samples = 10**4
        sample = np.zeros((n_samples,))
        for i in range(n_samples):
            sample[i] = bb_dist.sample_generalized_normal(mu=mu, alpha=alpha, p=p)
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

    def test_eval_uniform_logpdf_approx(self):
        # For a specific parameters test that the log pdf
        # approximation gives the correct values inside the bounds, and then
        # suppressed values outside the bounds.
        lower = -10
        upper = 10

        # Test within the bounds
        eval_at = np.linspace(-10,10,100)
        lpdf_approx = bb_dist.eval_uniform_logpdf_approx(eval_at,lower,upper)
        lpdf = bb_dist.eval_uniform_logpdf(eval_at,lower,upper)
        precision = 5
        np.testing.assert_almost_equal(lpdf_approx, lpdf, precision)

        # Test outside the bounds
        eval_at = np.linspace(-20,-10.0001,100)
        lpdf_approx = bb_dist.eval_uniform_logpdf_approx(eval_at,lower,upper)
        np.testing.assert_almost_equal(lpdf_approx,-990+eval_at-np.log(upper-lower),precision)

        eval_at = np.linspace(10.0001,20,100)
        lpdf_approx = bb_dist.eval_uniform_logpdf_approx(eval_at,lower,upper)
        np.testing.assert_almost_equal(lpdf_approx,-990-eval_at-np.log(upper-lower),precision)

    def test_eval_normal_logpdf_approx(self):
        # For a specific mu, sigma, upper, and lower, test that the log pdf
        # approximation gives the correct values inside the bounds, and then
        # suppressed values outside the bounds.
        mu = 1
        sigma = 5
        lower = -10
        upper = 10

        # Test within the bounds
        eval_at = np.linspace(-10,10,100)
        lpdf_approx = bb_dist.eval_normal_logpdf_approx(eval_at,mu,sigma,lower,upper)
        lpdf = bb_dist.eval_normal_logpdf(eval_at,mu,sigma,lower,upper)
        precision = 5
        np.testing.assert_almost_equal(lpdf_approx, lpdf, precision)

        # Test outside the bounds
        eval_at = np.linspace(-20,-10.0001,100)
        lpdf_approx = bb_dist.eval_normal_logpdf_approx(eval_at,mu,sigma,lower,upper)
        lpdf = bb_dist.eval_normal_logpdf(eval_at,mu,sigma)
        np.testing.assert_array_less(lpdf_approx, lpdf)
        # assert greater because of the accept_norm
        np.testing.assert_array_less(lpdf-1000,lpdf_approx)

        eval_at = np.linspace(10.0001,20,100)
        lpdf_approx = bb_dist.eval_normal_logpdf_approx(eval_at,mu,sigma,lower,upper)
        lpdf = bb_dist.eval_normal_logpdf(eval_at,mu,sigma)
        np.testing.assert_array_less(lpdf_approx, lpdf)
        # assert greater because of the accept_norm
        np.testing.assert_array_less(lpdf-1000,lpdf_approx)

        # Test that the default values work
        eval_at = np.linspace(-10,10,100)
        lpdf_approx = bb_dist.eval_normal_logpdf_approx(eval_at,mu,sigma)
        lpdf = bb_dist.eval_normal_logpdf(eval_at,mu,sigma)

    def test_eval_lognormal_logpdf_approx(self):
        # For a specific mu, sigma, upper, and lower, test that the log pdf
        # approximation gives the correct values inside the bounds, and then
        # suppressed values outside the bounds.
        mu = 1
        sigma = 5
        lower = 1
        upper = 10

        # Test within the bounds
        eval_at = np.linspace(1,10,100)
        lpdf_approx = bb_dist.eval_lognormal_logpdf_approx(eval_at,mu,sigma,lower,upper)
        lpdf = bb_dist.eval_lognormal_logpdf(eval_at,mu,sigma,lower,upper)
        precision = 5
        np.testing.assert_almost_equal(lpdf_approx, lpdf, precision)

        # Test outside the bounds
        eval_at = np.linspace(0.0000001,0.9999,100)
        lpdf_approx = bb_dist.eval_lognormal_logpdf_approx(eval_at,mu,sigma,lower,upper)
        lpdf = bb_dist.eval_lognormal_logpdf(eval_at,mu,sigma)
        np.testing.assert_array_less(lpdf_approx, lpdf)
        # assert greater because of the accept_norm
        np.testing.assert_array_less(lpdf-1000,lpdf_approx)

        eval_at = np.linspace(10.0001,20,100)
        lpdf_approx = bb_dist.eval_lognormal_logpdf_approx(eval_at,mu,sigma,lower,upper)
        lpdf = bb_dist.eval_lognormal_logpdf(eval_at,mu,sigma)
        np.testing.assert_array_less(lpdf_approx, lpdf)
        # assert greater because of the accept_norm
        np.testing.assert_array_less(lpdf-1000,lpdf_approx)

        # Check that without bounds the function behaves as expected.
        lpdf_approx = bb_dist.eval_lognormal_logpdf_approx(eval_at,mu,sigma)
        lpdf = bb_dist.eval_lognormal_logpdf(eval_at,mu,sigma)
        np.testing.assert_almost_equal(lpdf_approx, lpdf, precision)

        # Check that the function doesn't fail if the lower is set to -np.inf
        lpdf_approx = bb_dist.eval_lognormal_logpdf_approx(eval_at,mu,sigma,lower=-np.inf)
        lpdf = bb_dist.eval_lognormal_logpdf(eval_at,mu,sigma)
        np.testing.assert_almost_equal(lpdf_approx, lpdf, precision)


    def test_eval_beta_logpdf_approx(self):
        # For a specific parameters est that the log pdf
        # approximation gives the correct values inside the bounds, and then
        # suppressed values outside the bounds.
        a = 2
        b = 2
        lower = -10
        upper = 10
        epsilon = 1e-9

        # Test within the bounds
        eval_at = np.linspace(-10+epsilon,10-epsilon,100)
        lpdf_approx = bb_dist.eval_beta_logpdf_approx(eval_at,a,b,lower,upper)
        lpdf = bb_dist.eval_beta_logpdf(eval_at,a,b,lower,upper)
        precision = 5
        np.testing.assert_almost_equal(lpdf_approx, lpdf, precision)

        # Test outside the bounds
        eval_at = np.linspace(-20,-10.0001,100)
        lpdf = bb_dist.eval_beta_logpdf(-10+epsilon,a,b,lower,upper)
        lpdf_approx = bb_dist.eval_beta_logpdf_approx(eval_at,a,b,lower,upper)
        np.testing.assert_almost_equal(lpdf_approx, lpdf+eval_at-lower-epsilon,precision)

        eval_at = np.linspace(10.0001,20,100)
        lpdf = bb_dist.eval_beta_logpdf(10-epsilon,a,b,lower,upper)
        lpdf_approx = bb_dist.eval_beta_logpdf_approx(eval_at,a,b,lower,upper)
        np.testing.assert_almost_equal(lpdf_approx, lpdf-eval_at+upper-epsilon,precision)
    
    def test_eval_generalized_normal_logpdf_approx(self):
        # For a specific parameters test that the log pdf
        # approximation gives the correct values inside the bounds, and then
        # suppressed values outside the bounds.
        mu = 0
        alpha = 1
        p = 10.0
        lower = -1.5
        upper = 1.5

        # Test within the bounds
        eval_at = np.linspace(-1.5,1.5,100)
        lpdf_approx = bb_dist.eval_generalized_normal_logpdf_approx(eval_at,mu,alpha,p,lower,upper)
        lpdf = bb_dist.eval_generalized_normal_logpdf(eval_at,mu,alpha,p,lower,upper)
        precision = 5
        np.testing.assert_almost_equal(lpdf_approx, lpdf, precision)

        # Test outside the bounds
        eval_at = np.linspace(-2,-1.5001,100)
        lpdf_approx = bb_dist.eval_generalized_normal_logpdf_approx(eval_at,mu,alpha,p,lower,upper)
        lpdf = bb_dist.eval_generalized_normal_logpdf(eval_at,mu,alpha,p)
        np.testing.assert_array_less(lpdf_approx, lpdf)
        # assert greater because of the accept_norm
        np.testing.assert_almost_equal(lpdf-1000,lpdf_approx,precision)

        eval_at = np.linspace(1.5001,2,100)
        lpdf_approx = bb_dist.eval_generalized_normal_logpdf_approx(eval_at,mu,alpha,p,lower,upper)
        lpdf = bb_dist.eval_generalized_normal_logpdf(eval_at,mu,alpha,p)
        np.testing.assert_array_less(lpdf_approx, lpdf)
        # assert greater because of the accept_norm
        np.testing.assert_almost_equal(lpdf-1000,lpdf_approx,precision)

        mu = 0
        alpha = 1
        p = 10.0
        lower = -10
        upper = 10

        # Test within the bounds
        eval_at = np.linspace(-10,10,100)
        lpdf_approx = bb_dist.eval_generalized_normal_logpdf_approx(eval_at,mu,alpha,p,lower,upper)
        lpdf = bb_dist.eval_generalized_normal_logpdf(eval_at,mu,alpha,p,lower,upper)
        precision = 5
        np.testing.assert_almost_equal(lpdf_approx, lpdf, precision)

        mu = 0.01
        alpha = 1
        p = 10.0
        # Test it at the border of the cutoff
        lower = -np.power(59.999999999,1/p)*alpha+mu
        upper = np.power(59.999999999,1/p)*alpha+mu

        # Test within the bounds
        eval_at = np.linspace(lower,upper,100)
        lpdf_approx = bb_dist.eval_generalized_normal_logpdf_approx(eval_at,mu,alpha,p,lower,upper)
        lpdf = bb_dist.eval_generalized_normal_logpdf(eval_at,mu,alpha,p,lower,upper)
        precision = 5
        np.testing.assert_almost_equal(lpdf_approx, lpdf, precision)

        eval_at = np.linspace(-1.5,1.5,100)
        lpdf_approx = bb_dist.eval_generalized_normal_logpdf_approx(eval_at,mu,alpha,p)
        lpdf = bb_dist.eval_generalized_normal_logpdf(eval_at,mu,alpha,p)
        np.testing.assert_almost_equal(lpdf_approx, lpdf, precision)

if __name__ == '__main__':
    unittest.main()

