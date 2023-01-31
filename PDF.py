from scipy.stats.sampling import DiscreteAliasUrn, DiscreteGuideTable
from scipy.stats import norm, genexpon, lognorm, rice, rayleigh, chisquare
from scipy.special import ndtr, kv, kn, gamma
import numpy as np
from matplotlib import pyplot as plt

class distributions:
    def __init__(self):
        self.seed = int(abs(norm.rvs() * 10))
    def norm_pdf(self, var, bias=0.0, steps=0.0):
        x = np.linspace(0.0, norm.ppf(0.99, var), steps)
        pdf = norm.pdf(x, scale=var)
        return x, pdf
    def norm_rvs(self, var, bias=0.0, steps=0.0):
        return norm.rvs(size=steps, scale=var, random_state=np.random.default_rng())
    def norm_cdf(self, var, bias=0.0, steps=0.0):
        x = np.linspace(0.0, norm.ppf(0.99, var), steps)
        return norm.cdf(x, scale=var)

    def lognorm_pdf(self, var, bias, steps):
        x = np.linspace(0.0, lognorm.ppf(0.99, var), steps)
        pdf = lognorm.pdf(x, s=var, loc=bias)
        return x, pdf
    def lognorm_rvs(self, steps, var, bias):
        return lognorm.rvs(size=steps, s=var, loc=bias, random_state=np.random.default_rng())
    def lognorm_cdf(self, x, var, bias):
        return lognorm.cdf(x, scale=var, loc=bias)


    def gg_pdf(self, alpha, beta, steps):
        # x_alpha = np.linspace(gamma.ppf(0.01, alpha), gamma.ppf(0.99, alpha), steps)
        # x_beta  = np.linspace(gamma.ppf(0.01, alpha), gamma.ppf(0.99, alpha), steps)
        # x = x_alpha * x_beta
        x = np.linspace(0.0, 5.0, steps)

        k = (alpha + beta) / 2
        k1 = alpha * beta
        K = 2 * (k1**k) / (gamma(alpha) * gamma(beta))
        K1 = x**(k-1)
        Z = 2 * np.sqrt(k1 * x)
        bessel = kv(alpha - beta, Z)
        pdf = K * K1 * bessel
        return x, pdf

    def gg_rvs(self, pdf, steps):
        pv = pdf[1:] / np.sum(pdf[1:])
        rng = DiscreteAliasUrn(pv, random_state=np.random.default_rng())
        rvs = rng.rvs(size=steps)

        return rvs

    def plot_pdf(self, x, pdf, name="gamma-gamma"):
        fig, axs = plt.subplots(1,1)
        axs.set_title('PDF - ' + str(name))
        axs.plot(x, pdf)
        axs.set_xlabel('Normalized intensity [-]')
        axs.set_ylabel('Probability density [-]')


dist = distributions()