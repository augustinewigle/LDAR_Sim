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

def get_bayes_measured_rate(technology, params, true_leak):
    # get measured rate based on the bayesian model and true_leak
    if technology == "QOGI_A":
        error = np.random.normal(0, 1/(params['tau']+true_leak/params['eta']))
        return (params['alpha0'] + params['alpha1'] * true_leak + params['alpha2'] * true_leak**2) * np.exp(error)
    elif technology == "QOGI_B":
        error = np.random.normal(0, 1/params['tau'])
        if true_leak <= params['gamma']:
            return (params['alpha0'] + params['alpha1'] * true_leak + params['alpha2'] * true_leak**2) * np.exp(error)
        else:
            return (params['alpha0'] + params['beta0'] + (params['alpha1'] + params['beta1']) * true_leak) * np.exp(error)
    elif technology == 'QOGI_C':
        error = np.random.normal(0, 1/(params['tau']+true_leak/params['eta']))
        if true_leak <= params['gamma']:
            return (params['alpha0'] + params['alpha1'] * true_leak + params['alpha2'] * true_leak**2) * np.exp(error)
        else:
            return (params['alpha0'] + params['beta0'] + (params['alpha1'] + params['beta1']) * true_leak) * np.exp(error)
    elif technology == "Truck_TDLAS":
        error = np.random.normal(0, 1/(params['tau']+true_leak/params['eta']))
        return (params['alpha0'] + params['alpha1'] * true_leak) * np.exp(error)
    elif technology == "Aerial_TDLAS":
        error = np.random.normal(0, 1/(params['tau']+true_leak/params['eta']))
        return (params['alpha1'] * true_leak) * np.exp(error)
    elif technology == "Aerial_NIRHSI":
        error = np.random.normal(0, 1/params['tau'])
        return (params['alpha0'] + params['alpha1'] * true_leak) * np.exp(error)
    else:
        print("Error: Input valid detection technology")
        sys.exit()

def infer_true_rate(technology, leak_params, inference_params, measured_rate):
     # sample L leaks from prior
    if inference_params['prior_dist'] == 'lognorm': # sample from lognormal distribution with the inputted shape and scale param. loc assumed to be 0
        Q_l = np.array(lognorm.rvs(s=inference_params['prior_params'][0],scale = inference_params['prior_params'][1],size=inference_params['prior_size'])) # dont need to sort, DescrStatsW takes care of it
    elif inference_params['prior_dist'] == 'uniform': # sample from uniform distribution with end poitns a,b. Note the uniform dis defined by loc and scale params with end points [loc, loc + scale].
        Q_l = np.array(uniform.rvs(loc = inference_params['prior_params'][0], scale = inference_params['prior_params'][1]-inference_params['prior_params'][0],size=inference_params['prior_size'])) 
    else:
        print("This prior distribution is not supported. Must be either lognorm or uniform.")       
        sys.exit() 
    # get posterior distribution based on measurement technology
    if technology == 'QOGI_A':
        means = np.log(leak_params['alpha0'] + leak_params['alpha1'] * Q_l + leak_params['alpha2'] * np.square(Q_l))
        m_rate_probs = norm(loc=means, scale=1/(leak_params['tau']+Q_l/leak_params['eta'])).pdf([np.log(measured_rate)])
    elif technology == 'QOGI_B':
        means = np.log(leak_params['alpha0'] + leak_params['alpha1'] * Q_l + leak_params['alpha2'] * np.square(Q_l))
        means[leak_params['gamma'] > Q_l] = np.log(leak_params['alpha0'] + leak_params['beta0'] + (leak_params['alpha1'] + leak_params['beta1']) * Q_l[leak_params['gamma'] > Q_l])
        m_rate_probs = norm(loc=means, scale=1/(leak_params['tau'])).pdf([np.log(measured_rate)])
    elif technology == 'QOGI_C':
        means = np.log(leak_params['alpha0'] + leak_params['alpha1'] * Q_l + leak_params['alpha2'] * np.square(Q_l))
        means[leak_params['gamma'] > Q_l] = np.log(leak_params['alpha0'] + leak_params['beta0'] + (leak_params['alpha1'] + leak_params['beta1']) * Q_l[leak_params['gamma'] > Q_l])
        m_rate_probs = norm(loc=means, scale=1/(leak_params['tau']+Q_l/leak_params['eta'])).pdf([np.log(measured_rate)])
    elif technology == 'Truck_TDLAS':
        means = np.log(leak_params['alpha0'] + leak_params['alpha1'] * Q_l)
        m_rate_probs = norm(loc=means, scale=1/(leak_params['tau']+Q_l/leak_params['eta'])).pdf([np.log(measured_rate)])
    elif technology == 'Aerial_TDLAS':
        means = np.log(leak_params['alpha1'] * Q_l)
        m_rate_probs = norm(loc=means, scale=1/(leak_params['tau']+Q_l/leak_params['eta'])).pdf([np.log(measured_rate)])
    elif technology == 'Aerial_NIRHSI':
        means = np.log(leak_params['alpha0'] + leak_params['alpha1'] * Q_l)
        m_rate_probs = norm(loc=means, scale=1/(leak_params['tau'])).pdf([np.log(measured_rate)])
    else:
        print("Error: Input valid detection technology")
        sys.exit()
    weights = m_rate_probs / sum(m_rate_probs)
    if inference_params['q'] >= 1.0:
        m_rate = sum(np.multiply(weights,Q_l)) 
    else: 
        m_rate = DescrStatsW(Q_l, weights = weights).quantile(inference_params['q'],return_pandas=False)[0]
    return m_rate

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

    if self.config["bayesian"]["bayes"]:
        # average of mcmc coefficients inputted as a method param under sensor
        measurement_tech = self.config['bayesian']['measurement_technology']
        leak_params = {'alpha0': self.config['bayesian']['alpha0'], 
                       'alpha1': self.config['bayesian']['alpha1'],
                       'alpha2': self.config['bayesian']['alpha2'],
                       'beta0': self.config['bayesian']['beta0'],
                       'beta1': self.config['bayesian']['beta1'],
                       'gamma': self.config['bayesian']['gamma'],
                       'tau': self.config['bayesian']['tau'],
                       'eta': self.config['bayesian']['eta']} 
        infer_true = self.config['bayesian']['infer_true']
        inference_params = {'prior_dist':self.config['bayesian']['prior'],
                        'prior_params':self.config['bayesian']['prior_params'],
                        'prior_size':self.config['bayesian']['L'],
                        'q': self.config['bayesian']['quantile']}
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
                    m_rate = get_bayes_measured_rate(measurement_tech, leak_params, rate)
                    # if we want to infer true rate using bayesian model, do so by following the algorithm
                    if infer_true:
                        m_rate = infer_true_rate(measurement_tech, leak_params, inference_params, m_rate)
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
                    m_rate = get_bayes_measured_rate(measurement_tech, leak_params, leak['rate'])
                    if infer_true:
                        m_rate = infer_true_rate(measurement_tech, leak_params, inference_params, m_rate)
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
