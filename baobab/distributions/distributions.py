import inspect
import numpy as np
import scipy.stats as stats
import numba
from math import gamma, erf

dist_names = ['uniform', 'normal', 'lognormal', 'beta', 'generalized_normal',]
__all__ = ['sample_{:s}'.format(d) for d in dist_names]
__all__ += ['sample_uniform_vectorize','sample_lognormal_vectorize',
            'sample_normal_vectorize']
__all__ += ['sample_multivar_normal','sample_one_minus_rayleigh']
__all__ += ['eval_{:s}_pdf'.format(d) for d in dist_names]
__all__ += ['eval_{:s}_logpdf'.format(d) for d in dist_names]
__all__ += ['eval_{:s}_logpdf_approx'.format(d) for d in dist_names]
__all__ += ['hyperparams', 'sample_transformed_kappa_normal', 'sample_delta_function']

def sample_delta_function(value):
    return value

def sample_transformed_kappa_normal(mu, sigma):
    """Effectively sample kappa by sampling x, defined by 1/(1-kappa), from a normal dist

    Parameters
    ----------
    mu : float
    sigma : float

    """
    x = np.random.normal(mu, sigma)
    while ~np.isfinite(1.0 - 1.0/x): # kappa = 1 - 1/x
        x = np.random.normal(mu, sigma)
    return 1.0 - 1.0/x

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

def sample_uniform_vectorize(size, lower, upper):
    """Sample from a uniform distribution

    Parameters
    ----------
    size : int
        the number of samples to draw
    lower : float
        min value
    upper : float
        max value

    Returns
    -------
    float
        uniform sample

    """
    u = np.random.rand(size)
    sample = lower + (upper - lower)*u
    return sample

def eval_uniform_pdf(eval_at, lower, upper):
    """Evaluate the uniform PDF

    See `sample_uniform` for parameter definitions.

    """
    normed_eval_pdf = np.ones_like(eval_at)/(upper-lower)
    normed_eval_pdf[eval_at<lower] = 0
    normed_eval_pdf[eval_at>upper] = 0
    return normed_eval_pdf

def eval_uniform_logpdf(eval_at, lower, upper):
    """Evaluate the uniform log PDF

    See `sample_uniform` for parameter definitions.

    """
    normed_eval_logpdf = np.zeros_like(eval_at)-np.log(upper-lower)
    normed_eval_logpdf[eval_at<lower] = -np.inf
    normed_eval_logpdf[eval_at>upper] = -np.inf
    return normed_eval_logpdf

@numba.njit
def eval_uniform_logpdf_approx(eval_at, lower, upper):
    """Evaluate the uniform log PDF without -np.inf

    See `sample_uniform` for parameter definitions.

    """
    eval_logpdf = np.zeros_like(eval_at)-np.log(upper-lower)

    eval_shape = eval_at.shape
    eval_at = eval_at.reshape(-1)
    eval_logpdf=eval_logpdf.reshape(-1)
    for e_i in range(len(eval_at)):
        if eval_at[e_i] < lower:
            eval_logpdf[e_i] -= (lower-eval_at[e_i]) + 1000
        if eval_at[e_i] > upper:
            eval_logpdf[e_i] -= (eval_at[e_i]-upper) + 1000
    eval_logpdf=eval_logpdf.reshape(eval_shape)
    return eval_logpdf

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

def sample_normal_vectorize(size,mu, sigma, lower=-np.inf, upper=np.inf):
    """Samples from a normal distribution, optionally truncated

    Parameters
    ----------
    size : int
        the number of samples to draw
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
                             loc=mu, scale=sigma).rvs(size=size)
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

def sample_lognormal_vectorize(size, mu, sigma, lower=-np.inf, upper=np.inf):
    """Samples from a lognormal distribution, optionally truncated

    Parameters
    ----------
    size : int
        the number of samples to draw
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
                             loc=mu, scale=sigma).rvs(size=size))
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

@numba.njit
def _norm_cdf(bound,mu,sigma):
    """
    A helper function for eval_normal_logpdf_approx

    See `sample_normal` for parameter definitions.

    """
    return 0.5*erf((bound-mu)/(sigma*np.sqrt(2)))

@numba.njit
def eval_normal_logpdf_approx(eval_at, mu, sigma, lower=-np.inf, upper=np.inf):
    """Evaluate the normal pdf, optionally truncated without -np.inf

    See `sample_normal` for parameter definitions.

    """
    # First calculate the function without bounds
    norm = -np.log(sigma)-np.log(2*np.pi)/2
    eval_logpdf = -np.power((eval_at-mu)/sigma,2)/2+norm
    accept_norm = _norm_cdf(upper,mu,sigma) - _norm_cdf(lower,mu,sigma)

    # Now correct for the bounds if they are not -np.inf and np.inf
    # Note, reshaping must always be done regardless of bounds or numba will not compile
    eval_shape = eval_at.shape
    eval_at = eval_at.reshape(-1)
    eval_logpdf=eval_logpdf.reshape(-1)
    if lower > -np.inf and upper < np.inf:
        for e_i in range(len(eval_at)):
            if eval_at[e_i] < lower:
                eval_logpdf[e_i] -= 1000
            if eval_at[e_i] > upper:
                eval_logpdf[e_i] -= 1000
    eval_logpdf=eval_logpdf.reshape(eval_shape)
    return eval_logpdf - np.log(accept_norm)

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

@numba.njit
def _lognorm_cdf(bound,mu,sigma):
    """
    A helper function for eval_lognormal_logpdf_approx

    See `sample_normal` for parameter definitions.

    """
    return 0.5*erf((np.log(bound)-mu)/(np.sqrt(2)*sigma))

@numba.njit
def eval_lognormal_logpdf_approx(eval_at, mu, sigma, lower=0, upper=np.inf):
    """Evaluate the normal pdf, optionally truncated without -np.inf

    See `sample_normal` for parameter definitions.

    """
    # First calculate the distribution without the bounds
    norm = -np.log(sigma) - np.log(eval_at) - np.log(2*np.pi)/2
    eval_unnormed_logpdf = -np.square(np.log(eval_at)-mu)/(2*sigma**2)
    eval_unnormed_logpdf += norm

    # Stop cdf from crashing if lower bound is below 0
    if lower<0:
        lower=0

    accept_norm = _lognorm_cdf(upper,mu,sigma) - _lognorm_cdf(lower,mu,sigma)
    eval_normed_logpdf = eval_unnormed_logpdf - np.log(accept_norm)

    # Now correct for the bounds if they are not -np.inf and np.inf
    # Note, reshaping must always be done regardless of bounds or numba will not compile
    eval_shape = eval_at.shape
    eval_at = eval_at.reshape(-1)
    eval_normed_logpdf=eval_normed_logpdf.reshape(-1)
    if lower > -np.inf and upper < np.inf:
        for e_i in range(len(eval_at)):
            if eval_at[e_i] < lower:
                eval_normed_logpdf[e_i] -= 1000
            if eval_at[e_i] > upper:
                eval_normed_logpdf[e_i] -= 1000
    eval_normed_logpdf=eval_normed_logpdf.reshape(eval_shape)
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

@numba.njit
def _beta_log_pdf_numba(eval_at,a,b):
    """“
    A helper function for eval_beta_logpdf_approx

    See `sample_beta` for parameter definitions.

    """
    return np.log(eval_at)*(a-1)+np.log(1-eval_at)*(b-1)

@numba.njit
def eval_beta_logpdf_approx(eval_at,a,b,lower,upper):
    """Evaluate the beta pdf, scaled/shifted without -np.inf

    See `sample_beta` for parameter definitions.

    """

    # Terms we only want to calculate once
    norm = np.log(gamma(a+b)/(gamma(a)*gamma(b)))
    scale = upper - lower
    lscale = np.log(scale)

    # Epsilon parameter for approximation
    epsilon = 1e-9/scale

    # The evaluations
    eval_logpdf = _beta_log_pdf_numba((eval_at-lower)/scale,a,b)-lscale+norm
    stitch_upper = _beta_log_pdf_numba(1-epsilon,a,b)-lscale+norm
    stitch_lower = _beta_log_pdf_numba(epsilon,a,b)-lscale+norm

    # Now set the values outside the bounds to fall of exponentially rather than be -np.inf
    # Note, reshaping must always be done regardless of bounds or numba will not compile
    eval_shape = eval_at.shape
    eval_at = eval_at.reshape(-1)
    eval_logpdf=eval_logpdf.reshape(-1)
    # For loops are not a problem with numba
    if lower > -np.inf and upper < np.inf:
        for e_i in range(len(eval_at)):
            if eval_at[e_i] < lower+epsilon:
                eval_logpdf[e_i] = stitch_lower - np.abs(eval_at[e_i] - lower - epsilon)
            if eval_at[e_i] > upper-epsilon:
                eval_logpdf[e_i] = stitch_upper - np.abs(eval_at[e_i] - upper + epsilon)
    eval_logpdf=eval_logpdf.reshape(eval_shape)
    return eval_logpdf

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

@numba.njit
def _incomplete_gamma(s,x):
    """
    A helper function for eval_generalized_normal_logpdf_approx

    See `sample_generalized_normal` for parameter definitions.

    """
    # For large x, _incomplete_gamma assymptotes to 1. We do not want to
    # compute the series for large x, so we will use this approximation.
    if x > 60:
        return gamma(s)
    ksum = 0
    xexp = np.exp(-x)
    # Summing 150 terms for x<80 gets us to a converged regime
    for k in range(100):
        ksum += xexp*np.power(x,k)/gamma(s+k+1)
    return x**s * gamma(s) * ksum

@numba.njit
def _gen_norm_cdf(bound,mu,alpha,p):
    """
    A helper function for eval_generalized_normal_logpdf_approx

    See `sample_generalized_normal` for parameter definitions.

    """
    x = np.power(np.abs(bound-mu)/alpha,p)
    return np.sign(bound-mu)*_incomplete_gamma(1/p,x)/(2*gamma(1/p))

@numba.njit
def eval_generalized_normal_logpdf_approx(eval_at, mu=0.0, alpha=1.0, p=10.0, lower=-np.inf, upper=np.inf):
    """Evaluate the generalized normal pdf, scaled/shifted

    See `sample_generalized_normal` for parameter definitions.

    """
    norm = np.log(p) - np.log(2) - np.log(alpha) - np.log(gamma(1/p))
    unnormed_eval_logpdf = - np.power(np.abs(eval_at-mu)/alpha,p)
    unnormed_eval_logpdf += norm
    accept_norm = _gen_norm_cdf(upper,mu,alpha,p) - _gen_norm_cdf(lower,mu,alpha,p)
    normed_eval_logpdf = unnormed_eval_logpdf - np.log(accept_norm)

    # Now correct for the bounds if they are not -np.inf and np.inf
    # Note, reshaping must always be done regardless of bounds or numba will not compile
    eval_shape = eval_at.shape
    eval_at = eval_at.reshape(-1)
    normed_eval_logpdf=normed_eval_logpdf.reshape(-1)
    # For loops are not a problem with numba
    if lower > -np.inf and upper < np.inf:
        for e_i in range(len(eval_at)):
            if eval_at[e_i] < lower:
                normed_eval_logpdf[e_i] -= 1000
            if eval_at[e_i] > upper:
                normed_eval_logpdf[e_i] -= 1000
    normed_eval_logpdf=normed_eval_logpdf.reshape(eval_shape)

    return normed_eval_logpdf

# Define the dictionary of distributions and their ordered list of hyperparams
hyperparams = {}
for dist_name in dist_names:
    sampling_f = globals()['sample_{:s}'.format(dist_name)]
    hyperparams[dist_name] = inspect.getargspec(sampling_f).args