# ------------------------------------------------------------------------------
# Program:     The LDAR Simulator (LDAR-Sim)
# File:        methods.sensor.default
# Purpose:     Detect emissions with a single value MDL threshold.
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
from methods.funcs import measured_rate as get_measured_rate
#from methods.funcs import measured_rate_bayes as get_bayes_rate
from utils.attribution import update_tag
import numpy as np
from scipy.stats import lognorm
from scipy.stats import norm
from scipy.stats import uniform
from statsmodels.stats.weightstats import DescrStatsW
import sys

def detect_emissions(
    self,
    site,
    covered_leaks,
    covered_equipment_rates,
    covered_site_rate,
    site_rate,
    venting,
    equipment_rates,
):
    equip_measured_rates = []
    site_measured_rate = 0
    found_leak = False
    n_leaks = len(covered_leaks)
    missed_leaks_str = "{}_missed_leaks".format(self.config["label"])

    if self.config["sensor"]["bayes"]:
        # average of mcmc coefficients inputted as a method param under sensor
        alpha0 = self.config['sensor']['alpha0']
        alpha1 = self.config['sensor']['alpha1']
        alpha2 = self.config['sensor']['alpha2']
        beta0 = self.config['sensor']['beta0']
        beta1 = self.config['sensor']['beta1']
        gamma = self.config['sensor']['gamma']
        tau = self.config['sensor']['tau']
        eta = self.config['sensor']['eta']
        prior_dist = self.config['sensor']['prior']
        prior_params = self.config['sensor']['prior_params']
        prior_size = self.config['sensor']['L']
        q = self.config['sensor']['quantile']
        if self.config["measurement_scale"] == "site":
            if covered_site_rate > self.config["sensor"]["MDL"][0]:
                found_leak = True
                site_measured_rate = get_measured_rate(covered_site_rate, self.config["sensor"]["QE"])
            else:
                site[missed_leaks_str] += n_leaks
                self.timeseries[missed_leaks_str][self.state["t"].current_timestep] += n_leaks
        elif self.config["measurement_scale"] == "equipment":
            for rate in covered_equipment_rates:
                # Probability of detection is independent of measurement error, use true rate to compare to MDL
                if rate > self.config["sensor"]["MDL"][0]:
                    found_leak = True
                    # sample epsilon from normal distribution
                    # compute measured rate according to the mcmc coefficients 
                    error = np.random.normal(0, 1/tau)
                    measured_rate = (alpha0 + alpha1*rate) * np.exp(error)
                    # sample L leaks from prior
                    if prior_dist == 'lognorm': # sample from lognormal distribution with the inputted shape and scale param. loc assumed to be 0
                        Q_l = np.array(lognorm.rvs(s=prior_params[0],scale = prior_params[1],size=prior_size)) # dont need to sort, DescrStatsW takes care of it
                    elif prior_dist == 'uniform': # sample from uniform distribution with end poitns a,b. Note the uniform dis defined by loc and scale params with end points [loc, loc + scale].
                        Q_l = np.array(uniform.rvs(loc = prior_params[0], scale = prior_params[1]-prior_params[0],size=prior_size)) 
                    else:
                        print("This prior distribution is not supported. Must be either lognorm or uniform.")       
                        sys.exit()     
                    # since m_rate = C * e^eps follows log normal, log(m_rate) = ln(c) + eps follows normal with mean ln(c) + 0 and var = var(eps)
                    means = np.log(alpha0 + alpha1*Q_l)
                    m_rate_probs = norm(loc=means, scale=1/tau).pdf([np.log(measured_rate)])
                    weights = m_rate_probs / sum(m_rate_probs)
                    # if quantile specified, calculate it based on the empirical distribution, else just do mean
                    if q >= 1.0:
                        m_rate = sum(np.multiply(weights,Q_l)) 
                    else: 
                        m_rate = DescrStatsW(Q_l, weights = weights).quantile(q,return_pandas=False)[0]
                else:
                    m_rate = 0
                equip_measured_rates.append(m_rate)
                site_measured_rate += m_rate
            if not found_leak:
                site[missed_leaks_str] += n_leaks
                self.timeseries[missed_leaks_str][self.state["t"].current_timestep] += n_leaks

        elif self.config["measurement_scale"] == "component":
            # If measurement scale is a leak, all leaks will be tagged
            # ^ Not true anymore, there can still be measurement error with component scale i.e. OGI
            # maybe take in technology type as one of the inputs, chnage likelihood/variance accordingly
            for leak in covered_leaks:
                if leak['rate'] > self.config["sensor"]["MDL"][0]:
                    found_leak = True
                    # sample L leaks from prior
                    if prior_dist == 'lognorm': # sample from lognormal distribution with the inputted shape and scale param. loc assumed to be 0
                        Q_l = np.array(lognorm.rvs(s=prior_params[0],scale = prior_params[1],size=prior_size)) # dont need to sort, DescrStatsW takes care of it
                    elif prior_dist == 'uniform': # sample from uniform distribution with end poitns a,b. Note the uniform dis defined by loc and scale params with end points [loc, loc + scale].
                        Q_l = np.array(uniform.rvs(loc = prior_params[0], scale = prior_params[1]-prior_params[0],size=prior_size)) 
                    else:
                        print("This prior distribution is not supported. Must be either lognorm or uniform.")       
                        sys.exit()     
                    # sample epsilon from normal distribution
                    error = np.random.normal(0,1/(tau + leak['rate']/eta))
                    # compute measured rate according to mcmc coefficients
                    if leak['rate'] <= gamma:
                        measured_rate = (alpha0 + alpha1 * leak['rate'] + alpha2 * leak['rate']**2) * np.exp(error)
                        means = np.log(alpha0 + alpha1 * Q_l + alpha2 * np.square(Q_l))
                    else:
                        measured_rate = (alpha0 + beta0 + (alpha1 + beta1)*leak['rate']) * np.exp(error)
                        means = np.log(alpha0 + beta0 + (alpha1 + beta1)*Q_l)
                    m_rate_probs = norm(loc=means, scale=1/(tau+leak['rate']/eta)).pdf([np.log(measured_rate)])
                    weights = m_rate_probs / sum(m_rate_probs) # div by 0 error sometimes, why?
                    # if quantile specified, calculate it based on the empirical distribution, else just do mean
                    if q >= 1.0:
                        m_rate = sum(np.multiply(weights,Q_l)) 
                    else: 
                        m_rate = DescrStatsW(Q_l, weights = weights).quantile(q,return_pandas=False)[0]
                    #print(m_rate,measured_rate) #m_rate and measured_rate are clearly different!!! But result is the same as non bayes?!?
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
    else:
        if self.config["measurement_scale"] == "site":
            if covered_site_rate > self.config["sensor"]["MDL"][0]:
                found_leak = True
                site_measured_rate = get_measured_rate(covered_site_rate, self.config["sensor"]["QE"])
            else:
                site[missed_leaks_str] += n_leaks
                self.timeseries[missed_leaks_str][self.state["t"].current_timestep] += n_leaks
        elif self.config["measurement_scale"] == "equipment":
            for rate in covered_equipment_rates:
                '''m_rate = get_measured_rate(rate, self.config["sensor"]["QE"])
                if m_rate > self.config["sensor"]["MDL"][0]:
                    found_leak = True
                else:
                    m_rate = 0'''
                if rate > self.config['sensor']['MDL'][0]:
                    found_leak = True
                    m_rate = get_measured_rate(rate, self.config['sensor']['QE'])
                else:
                    m_rate = 0
                equip_measured_rates.append(m_rate)
                site_measured_rate += m_rate
            if not found_leak:
                site[missed_leaks_str] += n_leaks
                self.timeseries[missed_leaks_str][self.state["t"].current_timestep] += n_leaks

        elif self.config["measurement_scale"] == "component":
            # If measurement scale is a leak, all leaks will be tagged
            for leak in covered_leaks:
                if leak['rate'] > self.config['sensor']['MDL'][0]:
                    measured_rate = get_measured_rate(leak['rate'],self.config['sensor']['QE'])
                    found_leak = True
                    is_new_leak = update_tag(
                        leak,
                        measured_rate,
                        site,
                        self.timeseries,
                        self.state["t"],
                        self.config["label"],
                        self.id,
                        self.program_parameters,
                    )
                    # Add these leaks to the 'tag pool'
                    if is_new_leak:
                        site_measured_rate += measured_rate
                else:
                    site[missed_leaks_str] += 1

    # Put all necessary information in a dictionary to be assessed at end of day
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
