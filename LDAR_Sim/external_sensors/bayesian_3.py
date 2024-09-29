"""
Alternative measurement error implementation based on Bayesian Inference in Wigle et al 2024
"""
# ------------------------------------------------------------------------------
# Program:     The LDAR Simulator (LDAR-Sim)
# File:        external_sensors.bayesian_3
# Purpose:     Sensor which uses measurement error based on Bayesian models
#
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the MIT License as published
# by the Free Software Foundation, version 3.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# MIT License for more details.

# You should have received a copy of the MIT License
# along with this program.  If not, see <https://opensource.org/licenses/MIT>.
#
# ------------------------------------------------------------------------------
import numpy as np
from scipy.stats import lognorm
from scipy.stats import norm
from scipy.stats import uniform
from statsmodels.stats.weightstats import DescrStatsW
from utils.attribution import update_tag
from methods.funcs import measured_rate


def detect_emissions(
    self,
    site,
    covered_leaks,
    covered_equipment_rates,
    covered_site_rate,
    site_rate,
    venting,
    equipment_rates,
):  # pylint: disable=unused-argument
    """
    An alternative sensor with a static MDL that implements a Bayesian error for uncertainty quantification:

    measured rate = (alpha1*true_rate)*exp(error) where error ~ N(0, (tau+true_rate/eta)^{-1}) # Bridger

    Uses three inputs to determine the distribution:
        alpha1
        tau
        eta
    Supply through QE parameter, QE = [alpha1, tau, eta]

    Args:
        site (site obj): Site in which crew is working at
        covered_leaks (list): list of leak objects that can be detected by the crew
        covered_equipment_rates (list): list of equipment leak rates that can be
                                        detected by the crew
        covered_site_rate (float): total site emissions from leaks that are observable
                                    from a crew
        site_rate (float): total site emissions from leaks all leaks at site
        venting (float): total site emissions from venting
        equipment_rates (list): list of equipment leak rates for each equipment group

    Returns:
        site report (dict):
                site  (site obj): same as input
                leaks_present (list): same as covered leaks input
                site_true_rate (float): same as site_rate
                site_measured_rate (float): total emis from all leaks measured
                equip_measured_rates (list): total of all leaks measured for each equip group
                venting (float): same as input
                found_leak (boolean): Did the crew find at least one leak at the site


    """

    missed_leaks_str = "{}_missed_leaks".format(self.config["label"])
    equip_measured_rates = []
    site_measured_rate = 0
    found_leak = False
    n_leaks = len(covered_leaks)
    QE = self.config["sensor"]["QE"]

    alpha1 = QE[0]
    tau = QE[1]
    eta = QE[2]

    infer_true = self.config['bayesian']['infer_true']
    inference_params = {'prior_dist':self.config['bayesian']['prior'], # lognorm or uniform
                        'prior_params':self.config['bayesian']['prior_params'], # two parameters defining either lognorm or uniform distribution
                        'prior_size':self.config['bayesian']['L'], # size of prior sample to take
                        'q': self.config['bayesian']['quantile']} # quantile of interest
    if self.config["measurement_scale"] == "site":
        if covered_site_rate > self.config["sensor"]["MDL"][0]:
            found_leak = True
            error = np.random.normal(0, 1/np.sqrt(tau+covered_site_rate/eta)) 
            site_measured_rate = (alpha1 * covered_site_rate) * np.exp(error)
        else:
         site[missed_leaks_str] += n_leaks
        self.timeseries[missed_leaks_str][self.state["t"].current_timestep] += n_leaks
    elif self.config["measurement_scale"] == "equipment":
        for rate in covered_equipment_rates:
            # Probability of detection is independent of measurement error, use true rate to compare to MDL
            if rate > self.config["sensor"]["MDL"][0]:
                found_leak = True
                # compute measured rate according to the mcmc coefficients 
                error = np.random.normal(0, 1/np.sqrt(tau + rate/eta)) 
                m_rate = (alpha1 * rate) * np.exp(error)

                # if we want to infer true rate using bayesian model, do so with the following function
                if infer_true:
                     m_rate = infer_true_rate(QE, inference_params, m_rate)
            else:
                m_rate = 0
            equip_measured_rates.append(m_rate)
            site_measured_rate += m_rate
        if not found_leak:
            site[missed_leaks_str] += n_leaks
            self.timeseries[missed_leaks_str][self.state["t"].current_timestep] += n_leaks

    elif self.config["measurement_scale"] == "component":
        for leak in covered_leaks:
            if leak['rate'] > self.config["sensor"]["MDL"][0]:
                found_leak = True
                # compute measured rate according to the mcmc coefficients
                error = np.random.normal(0, 1/np.sqrt(tau + leak["rate"]/eta)) 
                m_rate = (alpha1 * leak['rate']) * np.exp(error)
                # if we want to infer true rate using bayesian model, do so with the following function
                if infer_true:
                    m_rate = infer_true_rate(QE, inference_params, m_rate)
                is_new_leak = update_tag(
                    leak,
                    m_rate,
                    site,
                    self.timeseries,
                    self.state["t"],
                    self.config["label"],
                    self.id,
                    self.program_parameters,
                    )
                    # Add these leaks to the 'tag pool'
                if is_new_leak:
                    site_measured_rate += m_rate
            else:
                site[missed_leaks_str] += 1
                self.timeseries[missed_leaks_str][self.state["t"].current_timestep] += 1
    site_dict = {
        "site": site,
        "leaks_present": covered_leaks,
        "site_true_rate": site_rate,
        "site_measured_rate": site_measured_rate,
        "equip_measured_rates": equip_measured_rates,
        "vent_rate": venting,
        "found_leak": found_leak,
    }
    return site_dict

def infer_true_rate(QE, inference_params, measured_rate):

    alpha1 = QE[0]
    tau = QE[1]
    eta = QE[2]

    # based on the measured rate and technology, infer the true rate by generating a posterior of true rate given measured rate
    # sample L leaks from prior

    if inference_params['prior_dist'] == 'lognorm': # sample from lognormal distribution with the inputted shape and scale param. loc assumed to be 0
        Q_l = np.array(lognorm.rvs(s=inference_params['prior_params'][0],scale = inference_params['prior_params'][1],size=inference_params['prior_size'])) # dont need to sort, DescrStatsW takes care of it
    elif inference_params['prior_dist'] == 'uniform': # sample from uniform distribution with end poitns a,b. Note the uniform dis defined by loc and scale params with end points [loc, loc + scale].
        Q_l = np.array(uniform.rvs(loc = inference_params['prior_params'][0], scale = inference_params['prior_params'][1]-inference_params['prior_params'][0],size=inference_params['prior_size'])) 
    else:
        print("This prior distribution is not supported. Must be either lognorm or uniform.")       
        sys.exit() 
    # calculate probably of observing the measured rate given the true rate is Q_l
    means = np.log(alpha1 * Q_l)
    m_rate_probs = norm(loc=means, scale=1/np.sqrt(tau+Q_l/eta)).pdf([np.log(measured_rate)])
    # now we have weighted sample of priors, with greater weights given to Q_l with greater probably of observing the measured rate
    weights = m_rate_probs / sum(m_rate_probs)
    # by default, calculate the mean of the posterior
    if inference_params['q'] >= 1.0:
        m_rate = np.mean(np.multiply(weights,Q_l)) 
    else: # use DescrStatsW to compute quantiles in the sample of priors and respective weights
        m_rate = DescrStatsW(Q_l, weights = weights).quantile(inference_params['q'],return_pandas=False)[0]
    return m_rate