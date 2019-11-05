import numpy as np
import scipy.stats as stats
import lenstronomy.Util.param_util as param_util
from abc import ABC, abstractmethod

class BaseBNNPrior(ABC):
    """Abstract base class equipped with PDF evaluation and sampling utility functions for various lens/source macromodels

    """
    def __init__(self):
        self.set_required_parameters()

    def set_required_parameters(self):
        """Defines a dictionary of the list of parameters (value) corresponding to each profile (key).

        The parameter names follow the lenstronomy convention.
        The dictionary will be updated as more profiles are supported.

        """
        params = dict(SPEMD=['center_x', 'center_y', 'gamma', 'theta_E', 'e1', 'e2'],
                          SHEAR_GAMMA_PSI=['gamma_ext', 'psi_ext'],
                          SERSIC_ELLIPSE=['magnitude', 'center_x', 'center_y', 'n_sersic', 'R_sersic', 'e1', 'e2'],
                          LENSED_POSITION=['magnitude'],
                          SOURCE_POSITION=['ra_source', 'dec_source', 'magnitude'],)
        setattr(self, 'params', params)

    def _raise_config_error(self, missing_key, parent_config_key, bnn_prior_class):
        """Convenience function for raising errors related to config values

        """
        raise ValueError("{:s} must be specified in the config inside {:s} for {:s}".format(missing_key,
                                                                                             parent_config_key,
                                                                                             bnn_prior_class))
    
    def sample_param(self, hyperparams):
        """Assigns a sampling distribution

        """
        # TODO: see if direct attribute call is quicker than string comparison
        dist = hyperparams.pop('dist')
        if dist == 'beta':
            return self.sample_beta(**hyperparams)
        elif dist == 'normal':
            return self.sample_normal(**hyperparams)
        elif dist == 'generalized_normal':
            return self.sample_generalized_normal(**hyperparams)
        elif dist == 'sample_one_minus_rayleigh':
            return self.sample_one_minus_rayleigh(**hyperparams)
        else:
            raise NotImplementedError

    def eval_param_pdf(self, eval_at, hyperparams):
        """Assigns and evaluates the PDF 

        """
        # TODO: see if direct attribute call is quicker than string comparison
        dist = hyperparams.pop('dist')
        if dist == 'beta':
            return self.eval_beta_pdf(eval_at, **hyperparams)
        elif dist == 'normal':
            return self.eval_normal_pdf(eval_at, **hyperparams)
        elif dist == 'generalized_normal':
            return self.eval_generalized_normal_pdf(eval_at, **hyperparams)
        else:
            raise NotImplementedError

    def sample_one_minus_rayleigh(self, scale, lower):
        """Samples from a Rayleigh distribution and gets one minus the value,
        often used for ellipticity modulus

        Parameters
        ----------
        scale : float
            scale of the Rayleigh distribution
        lower : float
            min allowed value of the one minus Rayleigh sample

        Returns
        -------
        float
            one minus the Rayleigh sample

        """
        q = 0.0
        while q < lower:
            q = 1.0 - np.random.rayleigh(scale, size=None)
        return q

    def sample_normal(self, mu, sigma, lower=-np.inf, upper=np.inf, log=False):
        """Samples from a normal distribution, optionally truncated

        Parameters
        ----------
        mu : float
            mean
        sigma : float
            standard deviation
        lower : float
            min value (default: -np.inf)
        upper : float
            max value (default: np.inf)
        log : bool
            is log-parameterized (default: False)
            if True, the mu and sigma are in dexes 

        Returns 
        -------
        float
            a sample from the specified normal

            """
        sample = stats.truncnorm((lower - mu)/sigma, (upper - mu)/sigma,
                                 loc=mu, scale=sigma).rvs()
        if log:
            sample = np.exp(sample)
        return sample

    def eval_normal_pdf(self, eval_at, mu, sigma, lower=-np.inf, upper=np.inf, log=False):
        """Evaluate the normal pdf, optionally truncated

        See `sample_normal` for parameter definitions.

        """
        if log:
            dist = stats.lognorm(scale=np.exp(mu), s=sigma, loc=0.0)
            eval_unnormed_pdf = dist.pdf(eval_at)
            accept_norm = dist.cdf(upper) - dist.cdf(lower)
            eval_normed_pdf = eval_unnormed_pdf/accept_norm
            return eval_normed_pdf
        else:
            dist = stats.truncnorm((lower - mu)/sigma, (upper - mu)/sigma, loc=mu, scale=sigma)
            eval_pdf = dist.pdf(eval_at)
            return eval_pdf

    def sample_multivar_normal(self, mu, cov_mat, is_log=None, lower=None, upper=None):
        """Samples from an N-dimensional normal distribution, optionally truncated

        An error will be raised if the cov_mat is not PSD.

        Parameters
        ----------
        mu : 1-D array_like, of length N
            mean
        cov_mat : 2-D array_like, of shape (N, N)
            symmetric, PSD matrix
        is_log : 1-D array_like, of length N where each element is bool
            whether each param is log-parameterized
        lower : None, float, or 1-D array_like, of length N
            min values (default: None)
        upper : None, float, or 1-D array_like, of length N
            max values (default: None)

        Returns
        -------
        float
            a sample from the specified N-dimensional normal

            """
        N = len(mu)
        sample = np.random.multivariate_normal(mean=mu, cov=cov_mat, check_valid='raise')

        # TODO: get the PDF, scaled for truncation
        # TODO: issue warning if significant portion of marginal PDF is truncated
        if (lower is not None) or (upper is not None):
            if not (len(lower) == N and len(upper) == N):
                raise ValueError("lower and upper bounds must have length (# of parameters)")
            lower = -np.inf if lower is None else lower
            upper = np.inf if upper is None else upper
            # Reject samples outside of bounds, repeat sampling until accepted
            while not np.all([np.greater(sample, lower), np.greater(upper, sample)]):
                sample = np.random.multivariate_normal(mean=mu, cov=cov_mat)
        
        if is_log is not None:
            sample[is_log] = np.exp(sample[is_log])

        return sample

    def sample_beta(self, a, b, lower=0.0, upper=1.0):
        """Samples from a beta distribution, scaled/shifted

        Parameters
        ----------
        a : float
            first beta parameter
        b : float
            second beta parameter
        lower : float
            min value (default: 0.0)
        upper : float
            max value (default: 1.0)

        Returns 
        -------
        float
            a sample from the specified beta
        
        """
        sample = np.random.beta(a, b)
        sample = sample*(upper - lower) + lower
        # TODO: check if same as
        # stats.beta(a=a, b=b, loc=lower, scale=upper-lower).rvs()
        return sample

    def eval_beta_pdf(self, eval_at, a, b, lower=0.0, upper=1.0):
        """Evaluate the beta pdf, scaled/shifted

        See `sample_beta` for parameter definitions.

        """
        dist = stats.beta(a=a, b=b, loc=lower, scale=upper-lower)
        eval_pdf = dist.pdf(eval_at)
        return eval_pdf

    def sample_generalized_normal(self, mu=0.0, alpha=1.0, p=10.0, lower=-np.inf, upper=np.inf):
        """Samples from a generalized normal distribution, optionally truncated

        Note
        ----
        Also called the exponential power distribution, this distribution converges
        pointwise to uniform as p --> infinity. To approximate a uniform between ``a`` and ``b``,
        define ``mu = 0.5*(a + b)`` and ``alpha=0.5*(b - a)``.
        For ``p=1``, it's identical to Laplace.
        For ``p=2``, it's identical to normal.
        See [1]_.

        Parameters
        ----------
        mu : float
            location (default: 0.0)
        alpha : float
            scale (default: 1.0)
        p : float
            shape (default: 10.0)
        lower : float
            min value (default: -np.inf)
        upper : float
            max value (default: np.inf)

        References
        ----------
        .. [1] `"Generalized normal distribution, Version 1" <https://en.wikipedia.org/wiki/Generalized_normal_distribution#Version_1>`_

        """
        generalized_normal = stats.gennorm(beta=p, loc=mu, scale=alpha)
        sample = generalized_normal.rvs()
        # Reject samples outside of bounds, repeat sampling until accepted
        while not np.all([np.greater(sample, lower), np.greater(upper, sample)]):
            sample = generalized_normal.rvs()
        return sample

    def eval_generalized_normal_pdf(self, eval_at, mu=0.0, alpha=1.0, p=10.0, lower=-np.inf, upper=np.inf):
        """Evaluate the generalized normal pdf, scaled/shifted

        See `sample_generalized_normal` for parameter definitions.

        """
        generalized_normal = stats.gennorm(beta=p, loc=mu, scale=alpha)
        unnormed_eval_pdf = generalized_normal.pdf(eval_at)
        accept_norm = generalized_normal.cdf(upper) - generalized_normal.cdf(lower)
        normed_eval_pdf = unnormed_eval_pdf/accept_norm
        return normed_eval_pdf

    @abstractmethod
    def sample(self):
        """Gets kwargs of sampled parameters to be passed to lenstronomy

        Overridden by subclasses.

        """
        return NotImplemented