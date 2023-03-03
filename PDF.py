from scipy.stats.sampling import DiscreteAliasUrn, DiscreteGuideTable
from scipy.stats import norm, genexpon, lognorm, rice, rayleigh, chisquare
from scipy.special import ndtr, kv, kn, gamma, j0, i0
import numpy as np
from matplotlib import pyplot as plt

from input import *

class distributions:
    def __init__(self):
        self.seed = int(abs(norm.rvs() * 10))
    # Normal distribution
    def norm_pdf(self, sigma, mean=0.0, steps=0.0):
        x = np.linspace(-angle_div, angle_div, steps)
        pdf = 1/np.sqrt(2 * np.pi * sigma**2) * np.exp(-((x - mean) / sigma)**2/2)
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
        x = np.linspace(0.0, angle_div, steps)
        mean = np.sqrt(mean**2 + mean**2)
        pdf = x / sigma ** 2 * np.exp(-(x**2 + mean**2) / (2 * sigma**2)) * i0(x * mean / sigma ** 2)
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

    def plot(self, ax, sigma, input, x, pdf, data, index, effect, name):
        if effect == "scintillation" or effect == "beam wander" or effect == "angle of arrival":
            ax[0].set_title('PDF & Histogram: ' + str(effect) + ', ' + str(name))
            for i in range(len(index)):
                if effect == "scintillation":
                    shape, loc, scale = lognorm.fit(data[i])
                    pdf_data = lognorm.pdf(x, shape, loc, scale)
                else:
                    loc, scale = rayleigh.fit(data[i])
                    pdf_data = rayleigh.pdf(x, loc, scale)

                ax[i].plot(x, pdf_data, label='pdf fitted to histogram', color='red')
                ax[i].hist(data[i], density=True, range=(x.min(), x.max()))
                ax[i].plot(x, pdf[i], label='pdf theory, std='+str(sigma[i])+', var (normal) ='+str(input[i]))
                ax[i].legend()
                ax[i].set_ylabel('Probability density')


        elif effect == "TX jitter" or effect == "RX jitter":
            ax.set_title('PDF & Histogram: ' + str(effect) + ', ' + str(name))
            shape, loc, scale = rice.fit(data)
            pdf_data = rice.pdf(x, shape, loc, scale)
            ax.plot(x, pdf_data, label='pdf fitted to histogram, std='+str(scale*1.0E6)+'urad, std (normal) ='+str(loc*1.0E6)+' urad')
            ax.hist(data, density=True, range=(x.min(), x.max()))
            ax.plot(x, pdf, label='pdf theory, std='+str(sigma*1.0E6)+'urad, std (input) ='+str(np.sqrt(input[0])*1.0E6)+' urad, mean (input) ='+str(np.sqrt(input[1])*1.0E6)+' urad')
            ax.legend()
            ax.set_ylabel('Probability density')

        # elif effect == "TX jitter" or effect == "RX jitter":
        #     data1 = (data[0] - angle_pe_t) * 1.0E6
        #     sigma = sigma * 1.0E6
        #     ax[0].set_title('PDF & Histogram, pointing jitter: ' + str(name))
        #     # Plot the sampled data
        #     ax[0].hist(data1, x, density=True)
        #     # Plot PDF function
        #     ax[0].plot(x, pdf, label='variance: ' + str(np.round(var, 2)))
        #     ax[0].set_ylabel('Probability density [-]')
        #     ax[0].set_xlabel('Pointing jitter (x-axis) [urad]')
        #
        #     # Plot PDF function that fits to the sampled data (this should be the same as the other PDF)
        #     shape, loc, scale = lognorm.fit(data1)
        #     pdf_data = lognorm.pdf(x, shape, loc, scale)
        #     ax[0].plot(x, pdf_data, label='pdf fitted to hist', color='red')
        #     ax[0].legend()
        #
        #     ax[1].scatter(data[0]* 1.0E6, data[1]* 1.0E6)
        #     ax[1].scatter(np.mean(data[0]* 1.0E6), np.mean(data[1]* 1.0E6), label='Mean error (x: '+ str(np.round(np.mean(data[0]* 1.0E6),2))+', y: '+ str(np.round(np.mean(data[0]* 1.0E6),2))+' [urad])')
        #     ax[1].set_title(f'Pointing jitter angle (both axis): ' + str(PDF_pointing))
        #     ax[1].set_ylabel('Y-axis angle error + jitter [urad]')
        #     ax[1].set_xlabel('X-axis angle error + jitter [urad]')
        #     ax[1].legend()

        if effect == "scintillation":
            ax[-1].set_xlabel('Normalized power [P/P0]')
        elif effect == "beam wander" or effect == "angle of arrival":
            ax[-1].set_xlabel('Angular displacement [rad]')
        else:
            ax.set_xlabel('Angular displacement [rad]')



dist = distributions()