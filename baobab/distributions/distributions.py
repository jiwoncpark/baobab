import inspect
import numpy as np
import scipy.stats as stats

dist_names = ['uniform', 'normal', 'lognormal', 'beta', 'generalized_normal',]
__all__ = ['sample_{:s}'.format(d) for d in dist_names] 
__all__ += ['sample_multivar_normal','sample_one_minus_rayleigh']
__all__ += ['eval_{:s}_pdf'.format(d) for d in dist_names]
__all__ += ['eval_{:s}_logpdf'.format(d) for d in dist_names]
__all__ += ['hyperparams']

def sample_uniform(lower, upper):
    """Sample from a uniform distribution

    Parameters
    ----------
    lower : float
        min value
    upper : float
        max value

    Returns
    -------
    float
        uniform sample

    """
    u = np.random.rand()
    sample = lower + (upper - lower)*u
    return sample

def eval_uniform_pdf(eval_at, lower, upper):
    """Evaluate the uniform PDF

    See `sample_uniform` for parameter definitions.

    """
    return np.ones_like(eval_at)/(upper-lower)

def eval_uniform_logpdf(eval_at, lower, upper):
    """Evaluate the uniform log PDF

    See `sample_uniform` for parameter definitions.

    """
    return -np.log(upper-lower)

def sample_one_minus_rayleigh(scale, lower):
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

def sample_normal(mu, sigma, lower=-np.inf, upper=np.inf):
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

    Returns 
    -------
    float
        a sample from the specified normal

    """
    sample = stats.truncnorm((lower - mu)/sigma, (upper - mu)/sigma,
                             loc=mu, scale=sigma).rvs()
    return sample

def sample_lognormal(mu, sigma, lower=-np.inf, upper=np.inf):
    """Samples from a lognormal distribution, optionally truncated

    Parameters
    ----------
    mu : float
        mean in dexes
    sigma : float
        standard deviation in dexes
    lower : float
        min value (default: -np.inf)
    upper : float
        max value (default: np.inf)

    Returns 
    -------
    float
        a sample from the specified normal

    """
    sample = np.exp(stats.truncnorm((lower - mu)/sigma, (upper - mu)/sigma,
                             loc=mu, scale=sigma).rvs())
    return sample

def eval_normal_pdf(eval_at, mu, sigma, lower=-np.inf, upper=np.inf):
    """Evaluate the normal pdf, optionally truncated

    See `sample_normal` for parameter definitions.

    """
    dist = stats.truncnorm((lower - mu)/sigma, (upper - mu)/sigma, loc=mu, scale=sigma)
    eval_pdf = dist.pdf(eval_at)
    return eval_pdf

def eval_normal_logpdf(eval_at, mu, sigma, lower=-np.inf, upper=np.inf):
    """Evaluate the normal pdf, optionally truncated

    See `sample_normal` for parameter definitions.

    """
    dist = stats.truncnorm((lower - mu)/sigma, (upper - mu)/sigma, loc=mu, scale=sigma)
    eval_pdf = dist.logpdf(eval_at)
    return eval_pdf

def eval_lognormal_pdf(eval_at, mu, sigma, lower=-np.inf, upper=np.inf):
    """Evaluate the normal pdf, optionally truncated

    See `sample_normal` for parameter definitions.

    """
    dist = stats.lognorm(scale=np.exp(mu), s=sigma, loc=0.0)
    eval_unnormed_pdf = dist.pdf(eval_at)
    accept_norm = dist.cdf(upper) - dist.cdf(lower)
    eval_normed_pdf = eval_unnormed_pdf/accept_norm
    eval_unnormed_pdf[eval_at<lower] = 0
    eval_unnormed_pdf[eval_at>upper] = 0
    return eval_normed_pdf

def eval_lognormal_logpdf(eval_at, mu, sigma, lower=-np.inf, upper=np.inf):
    """Evaluate the normal pdf, optionally truncated

    See `sample_normal` for parameter definitions.

    """
    dist = stats.lognorm(scale=np.exp(mu), s=sigma, loc=0.0)
    eval_unnormed_logpdf = dist.logpdf(eval_at)
    accept_norm = dist.cdf(upper) - dist.cdf(lower)
    eval_normed_logpdf = eval_unnormed_logpdf - np.log(accept_norm)
    eval_unnormed_logpdf[eval_at<lower] = -np.inf
    eval_unnormed_logpdf[eval_at>upper] = -np.inf
    return eval_normed_logpdf

def sample_multivar_normal(mu, cov_mat, is_log=None, lower=-np.inf, upper=np.inf):
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
        # Reject samples outside of bounds, repeat sampling until accepted
        while not np.all([np.greater(sample, lower), np.greater(upper, sample)]):
            print(sample)
            print(lower, upper)
            sample = np.random.multivariate_normal(mean=mu, cov=cov_mat)
    
    if is_log is not None:
        sample[is_log] = np.exp(sample[is_log])

    return sample

def sample_beta(a, b, lower=0.0, upper=1.0):
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

def eval_beta_pdf(eval_at, a, b, lower=0.0, upper=1.0):
    """Evaluate the beta pdf, scaled/shifted

    See `sample_beta` for parameter definitions.

    """
    dist = stats.beta(a=a, b=b, loc=lower, scale=upper-lower)
    eval_pdf = dist.pdf(eval_at)
    return eval_pdf

def eval_beta_logpdf(eval_at, a, b, lower=0.0, upper=1.0):
    """Evaluate the beta pdf, scaled/shifted

    See `sample_beta` for parameter definitions.

    """
    dist = stats.beta(a=a, b=b, loc=lower, scale=upper-lower)
    eval_pdf = dist.logpdf(eval_at)
    return eval_pdf

def sample_generalized_normal(mu=0.0, alpha=1.0, p=10.0, lower=-np.inf, upper=np.inf):
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
    sample = stats.gennorm.rvs(size=None, beta=p, loc=mu, scale=alpha)
    # Reject samples outside of bounds, repeat sampling until accepted
    out_of_bounds = not np.logical_and(np.greater(sample, lower), np.greater(upper, sample))
    #print(out_of_bounds, sample, lower, upper)
    while out_of_bounds is True:
        sample = stats.gennorm.rvs(size=None, beta=p, loc=mu, scale=alpha)
        out_of_bounds = not np.logical_and(np.greater(sample, lower), np.greater(upper, sample))
    return sample

def eval_generalized_normal_pdf(eval_at, mu=0.0, alpha=1.0, p=10.0, lower=-np.inf, upper=np.inf):
    """Evaluate the generalized normal pdf, scaled/shifted

    See `sample_generalized_normal` for parameter definitions.

    """
    generalized_normal = stats.gennorm(beta=p, loc=mu, scale=alpha)
    unnormed_eval_pdf = generalized_normal.pdf(eval_at)
    unnormed_eval_pdf[eval_at<lower] = 0
    unnormed_eval_pdf[eval_at>upper] = 0
    accept_norm = generalized_normal.cdf(upper) - generalized_normal.cdf(lower)
    normed_eval_pdf = unnormed_eval_pdf/accept_norm
    return normed_eval_pdf

def eval_generalized_normal_logpdf(eval_at, mu=0.0, alpha=1.0, p=10.0, lower=-np.inf, upper=np.inf):
    """Evaluate the generalized normal pdf, scaled/shifted

    See `sample_generalized_normal` for parameter definitions.

    """
    generalized_normal = stats.gennorm(beta=p, loc=mu, scale=alpha)
    unnormed_eval_logpdf = generalized_normal.logpdf(eval_at)
    unnormed_eval_logpdf[eval_at<lower] = -np.inf
    unnormed_eval_logpdf[eval_at>upper] = -np.inf
    accept_norm = generalized_normal.cdf(upper) - generalized_normal.cdf(lower)
    normed_eval_logpdf = unnormed_eval_logpdf - np.log(accept_norm)
    return normed_eval_logpdf

# Define the dictionary of distributions and their ordered list of hyperparams 
hyperparams = {}
for dist_name in dist_names:
    sampling_f = globals()['sample_{:s}'.format(dist_name)]
    hyperparams[dist_name] = inspect.getargspec(sampling_f).args