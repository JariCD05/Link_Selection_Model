from scipy.stats.sampling import DiscreteAliasUrn, DiscreteGuideTable
from scipy.stats import norm, genexpon, lognorm, rice, rayleigh, chisquare, beta, rv_histogram
from scipy.special import ndtr, kv, kn, gamma, j0, i0
import numpy as np
from matplotlib import pyplot as plt

from helper_functions import *

from input import *

class distributions:
    def __init__(self):
        self.seed = int(abs(norm.rvs() * 10))
    # Normal distribution
    def norm_pdf(self, sigma, mean=0.0, steps=0.0):
        x = np.linspace(-angle_div, angle_div, steps)
        pdf = 1/np.sqrt(2 * np.pi * sigma**2) * np.exp(-1/2 * ((x - mean) / sigma)**2)
        return x, pdf
    def norm_rvs(self, data, sigma, mean):
        return sigma * data + mean

    # Log-normal distribution
    def lognorm_pdf(self, sigma, mean, steps):
        x = np.linspace(0.0, 3.0, steps)
        pdf = 1 / (sigma * x * np.sqrt(2*np.pi)) * np.exp(-(np.log(x)- mean)**2/(2 * sigma**2))
        return x, pdf
    def lognorm_rvs(self, data, sigma, mean):
        return np.exp(mean + sigma * data)

    # Rayleigh distribution
    def rayleigh_pdf(self, sigma, steps):
        x = np.linspace(0.0, angle_div, steps)
        pdf = x / sigma**2 * np.exp(-x**2 / (2 * sigma**2))
        return x, pdf

    def rayleigh_rvs(self, data, sigma):
        # REF: Power vector generation tool for free-space optical links - PVGeT, Giggenbach, FIG.3
        return sigma * np.sqrt(data[0]**2 + data[1]**2)

    # Rician (Rice) distribution
    def rice_pdf(self, sigma, mean, steps):
        x = np.linspace(0.0, angle_div/1.5, steps)
        pdf = x / sigma ** 2 * np.exp(-(x**2 + mean**2) / (2 * sigma**2)) * i0(x * mean / sigma ** 2)
        return x, pdf

    def beta_pdf(self, sigma, steps):
        x = np.linspace(0, 1, steps)
        beta = w0 ** 2 / (4 * sigma ** 2)
        pdf = beta * x ** (beta - 1)
        return x, pdf


    # Gamma Gamma distribution
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

    def plot(self, ax, sigma, mean, x, pdf, data, index, effect, name):
        range_x = x.max() - x.min()
        samples = len(x)
        number_of_intervals = int(np.sqrt(samples))
        number_of_intervals_norm = int(np.sqrt(samples))
        width_of_intervals = range_x / number_of_intervals
        if effect == "scintillation" or effect == "beam wander" or effect == "angle of arrival":
            ax[0].set_title('PDF & Histogram: ' + str(effect) + ', ' + str(name))
            for i in range(len(index)):
                if effect == "scintillation":
                    sigma_I = np.var(data[i]) / np.mean(data[i])**2 - 1
                    sigma_I_theory = np.exp(-2 * mean) - 1
                    # sigma_I_theory = sigma**2

                    # Create histogram parameters
                    shape, loc, scale = lognorm.fit(data[i])
                    sigma_hist = lognorm.std(s=shape, loc=loc, scale=scale)
                    # Convert to lognormal parameters
                    sigma_hist = np.sqrt(np.log(sigma_hist ** 2 + 1))
                    mean_hist = -0.5 * sigma_hist ** 2

                    pdf_data = lognorm.pdf(x, shape, loc, scale)
                    # PDF, fitted to histogram
                    ax[i].hist(data[i], density=True, bins=1000, range=(x.min(), x.max()))
                    ax[i].plot(x, pdf_data, label='pdf fitted to hist., '
                                                  '$\sigma$=' + str(np.round(sigma_hist, 3)) + ', '
                                                  '$\mu$=' + str(np.round(mean_hist, 3))+', $\sigma_I$='+str(sigma_I_theory[i]), color='red')
                    # Theoretical PDF
                    ax[i].plot(x, pdf[i], label='pdf theory, '
                                                '$\sigma$=' + str(np.round(sigma[i], 3)) + ', '
                                                '$\mu$=' + str(np.round(mean[i], 3)))
                else:
                    loc, scale = rayleigh.fit(data[i])
                    pdf_data = rayleigh.pdf(x=x, loc=loc, scale=scale)
                    mean_hist = np.mean(data[i])

                    # PDF, fitted to histogram
                    ax[i].hist(data[i], density=True, bins=1000, range=(x.min(), x.max()))
                    ax[i].plot(x, pdf_data, label='pdf fitted to hist., '
                                                  '$\sigma$='+ str(np.round(scale*1.0E6,3))+'urad, '
                                                  '$\mu$=' + str(np.round(mean_hist*1.0E6,3))+'urad', color='red')
                    # Theoretical PDF
                    ax[i].plot(x, pdf[i], label='pdf theory, '
                                                '$\sigma$=' + str(np.round(sigma[i]*1.0E6,3))+'urad, '
                                                '$\mu$=' + str(np.round(mean[i]*1.0E6,3))+'urad')

                ax[i].legend()
                ax[i].set_ylabel('Probability density')


        elif effect == "TX jitter" or effect == "RX jitter":
            ax.set_title('PDF & Histogram: ' + str(effect) + ', ' + str(name))
            # Create histogram parameters
            loc, scale = rayleigh.fit(data)
            mean_hist = np.mean(data)

            pdf_data = rayleigh.pdf(x=x, loc=loc, scale=scale)
            # PDF, fitted to histogram
            ax.hist(data, density=True, bins=1000, range=(x.min(), x.max()))
            ax.plot(x, pdf_data, label='pdf fitted to histogram, '
                                       '$\sigma$='+str(np.round(scale*1.0E6,3))+'urad, '
                                       '$\mu$='+str(np.round(mean_hist*1.0E6,3)), color='red')
            # Theoretical PDF
            ax.plot(x, pdf, label='pdf theory, '
                                  '$\sigma$='+str(np.round(sigma*1.0E6,3))+'urad, '
                                  '$\mu$='+str(np.round(mean*1.0E6,3))+'urad')
            ax.legend()
            ax.set_ylabel('Probability density')


        elif effect == "combined":
            ax.set_title('PDF & Histogram: ' + str(effect) + ', ' + str(name))
            # Create histogram parameters
            hist = np.histogram(data, bins=1000)
            rv  = rv_histogram(hist, density=False)
            pdf_data = rv.pdf(x)

            # shape, loc, scale = beta.fit(data)
            # pdf_data = beta.pdf(x=x, shape=shape, loc=loc, scale=scale)
            # mean_hist = beta.std(pdf_data)
            # sig_hist = beta.mean(pdf_data)
            ax.hist(data, density=True, bins=1000, range=(x.min(), x.max()))
            ax.plot(x, pdf_data, label='pdf fitted to histogram, $\sigma$=' + str(np.round(sigma * 1.0E6, 3)) + 'urad, $\mu$=' + str(np.round(mean * 1.0E6, 3)), color='red')
            # Theoretical PDF
            ax.plot(x, pdf, label='pdf theory, $\sigma$=' + str(np.round(sigma * 1.0E6, 3)) + 'urad, $\mu$=' + str(np.round(mean * 1.0E6, 3)) + 'urad')
            ax.legend()
            ax.set_ylabel('Probability density')


        if effect == "scintillation" or effect == "combined":
            ax[-1].set_xlabel('Normalized intensity [I/I0]')
        elif effect == "beam wander" or effect == "angle of arrival":
            ax[-1].set_xlabel('Angular displacement [rad]')
        plt.show()


dist = distributions()