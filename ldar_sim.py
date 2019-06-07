import pandas as pd
import numpy as np
import csv
import os
import datetime
import sys
import random
from operator_agent import *
from OGI_company import *
from M21_company import *
from OGI_FU_company import *
from aircraft_company import *
from truck_company import *
from plotter import *
from daylight_calculator import *

class ldar_sim:
    def __init__ (self, state, parameters, timeseries):
        '''
        Construct the simulation.
        '''

        self.state = state
        self.parameters = parameters
        self.timeseries = timeseries

        # Read in the sites as a list of dictionaries
        print('Initializing sites...')
        with open(self.parameters['infrastructure_file']) as f:
            self.state['sites'] = [{k: v for k, v in row.items()}
                    for row in csv.DictReader(f, skipinitialspace=True)]
        
        # Shuffle all the entries to randomize order for identical 't_Since_last_LDAR' values
        random.shuffle(self.state['sites'])
            
        # Additional variable(s) for each site
        for site in self.state['sites']:
            site.update( {'total_emissions_kg': 0})
            site.update( {'active_leaks': 0})
            site.update( {'repaired_leaks': 0})
            site.update( {'lat_index': min(range(len(self.state['weather'].latitude)), 
                           key=lambda i: abs(self.state['weather'].latitude[i]-float(site['lat'])))})
            site.update( {'lon_index': min(range(len(self.state['weather'].longitude)), 
                           key=lambda i: abs(self.state['weather'].longitude[i]-float(site['lon'])%360))})
            
            # Check to make sure site is within range of grid-based data
            if float(site['lat']) > max(self.state['weather'].latitude):
                sys.exit('Simulation terminated: One or more sites is too far North and is outside the spatial bounds of your weather data!')
            if float(site['lat']) < min(self.state['weather'].latitude):
                sys.exit('Simulation terminated: One or more sites is too far South and is outside the spatial bounds of your weather data!')
            if float(site['lon'])%360 > max(self.state['weather'].longitude):
                sys.exit('Simulation terminated: One or more sites is too far East and is outside the spatial bounds of your weather data!')
            if float(site['lon'])%360 < min(self.state['weather'].longitude):
                sys.exit('Simulation terminated: One or more sites is too far West and is outside the spatial bounds of your weather data!')

        # Initialize method(s) to be used; append to state
        for m in self.parameters['methods']:
            if m == 'OGI':
                self.state['methods'].append (OGI_company (self.state,
                    self.parameters, self.parameters['methods'][m], timeseries))
            elif m == 'M21':
                self.state['methods'].append (M21_company (self.state,
                    self.parameters, self.parameters['methods'][m], timeseries))
            elif m == 'OGI_FU':
                self.state['methods'].append (OGI_FU_company (self.state,
                    self.parameters, self.parameters['methods'][m], timeseries))
            elif m == 'aircraft':
                self.state['methods'].append (aircraft_company (self.state,
                    self.parameters, self.parameters['methods'][m], timeseries))
            elif m == 'truck':
                self.state['methods'].append (truck_company (self.state,
                    self.parameters, self.parameters['methods'][m], timeseries))
            else:
                print ('Cannot add this method: ' + m)

        # Initialize baseline leaks for each site
        # First, generate initial leak count for each site
        print('Initializing leaks...')
        for site in self.state['sites']:
            n_leaks = round(np.random.normal(self.parameters['leaks_per_site_mean'], self.parameters['leaks_per_site_std']))
            if n_leaks <= 0:
                site.update({'initial_leaks': 0})
                self.state['init_leaks'].append(site['initial_leaks'])
            else:
                site.update({'initial_leaks': n_leaks})
                self.state['init_leaks'].append(site['initial_leaks'])

        # Second, load empirical leak-size data, switch from pandas to numpy (for speed), and convert g/s to kg/day
        self.empirical_leaks = pd.read_csv(self.parameters['leak_file'])
        self.empirical_leaks = np.array (self.empirical_leaks.iloc [:, 0])*84.
        self.state['max_rate'] = max(self.empirical_leaks)

        # Third, for each leak, create a dictionary and populate values for relevant keys
        for site in self.state['sites']:
            if site['initial_leaks'] > 0:
                for leak in range(site['initial_leaks']):
                    self.state['leaks'].append({
                                                'leak_ID': site['facility_ID'] + '_' + str(len(self.state['leaks']) + 1).zfill(10),
                                                'facility_ID': site['facility_ID'],
                                                'rate': self.empirical_leaks[np.random.randint(0, len(self.empirical_leaks))],
                                                'status': 'active',
                                                'days_active': 0,
                                                'component': 'unknown',
                                                'date_began': self.state['t'].current_date,
                                                'date_found': None,
                                                'date_repaired': None,
                                                'repair_delay': None,
                                                'found_by_company': None,
                                                'found_by_crew': None,
                                                'requires_shutdown': False,
                                                })

        # Initialize operator
        self.state['operator'] = operator_agent (self.timeseries, self.parameters, self.state)
        
        # Initialize daylight 
        if self.parameters['consider_daylight'] == True:
            self.state['daylight'] = daylight_calculator_ave(self.state, self.parameters)
            
        # Initialize empirical distribution of vented emissions
        if self.parameters['consider_venting'] == True:
        
        # Load empirical site emissions data, switch from pandas to numpy (for speed), and convert g/s to kg/day
            self.empirical_site = pd.read_csv(self.parameters['vent_file'])
            self.empirical_site = np.array (self.empirical_site.iloc [:, 0])*84.
            
        # Run Monte Carlo simulations to get distribution of vented emissions
            for i in range(1000):
                n_MC_leaks = round(np.random.normal(self.parameters['leaks_per_site_mean'], self.parameters['leaks_per_site_std']))
                MC_leaks = []
                for leak in range(n_MC_leaks):
                    MC_leaks.append(self.empirical_leaks[np.random.randint(0, len(self.empirical_leaks))])
                MC_leak_total = sum(MC_leaks)
                MC_site_total = self.empirical_site[np.random.randint(0, len(self.empirical_site))]
                MC_vent_total = MC_site_total - MC_leak_total
                self.state['empirical_vents'].append(MC_vent_total)
            
            # Change negatives to zero
            self.state['empirical_vents'] = [0 if i < 0 else i for i in self.state['empirical_vents']]
            
        return

    def update (self):
        '''
        this rolls the model forward one timestep
        returns nothing
        '''

        self.update_state()                 # Update state of sites and leaks
        self.add_leaks ()                   # Add leaks to the leak pool
        self.find_leaks ()                  # Find leaks
        self.repair_leaks ()                # Repair leaks
        self.report ()                      # Assemble any reporting about model state
        return

    def update_state (self):
        '''
        update the state of active leaks
        '''
        for leak in self.state['leaks']:
            if leak['status'] == 'active':
                leak['days_active'] += 1

        self.active_leaks = []
        for leak in self.state['leaks']:
            if leak['status'] == 'active':
                self.active_leaks.append(leak)
        self.timeseries['active_leaks'].append(len(self.active_leaks))
        self.timeseries['datetime'].append(self.state['t'].current_date)

    def add_leaks (self):
        '''
        add new leaks to the leak pool
        '''
        # First, determine whether each site gets a new leak or not
        for site in self.state['sites']:
            n_leaks = np.random.binomial(1, self.parameters['LPR'])
            if n_leaks == 0:
                site.update({'n_new_leaks': 0})
            else:
                site.update({'n_new_leaks': n_leaks})

        # For each leak, create a dictionary and populate values for relevant keys
        for site in self.state['sites']:
            if site['n_new_leaks'] > 0:
                for leak in range(site['n_new_leaks']):
                    self.state['leaks'].append({
                                                'leak_ID': site['facility_ID'] + '_' + str(len(self.state['leaks']) + 1).zfill(10),
                                                'facility_ID': site['facility_ID'],
                                                'rate': self.empirical_leaks[np.random.randint(0, len(self.empirical_leaks))],
                                                'status': 'active',
                                                'days_active': 0,
                                                'component': 'unknown',
                                                'date_began': self.state['t'].current_date,
                                                'date_found': None,
                                                'date_repaired': None,
                                                'repair_delay': None,
                                                'found_by_company': None,
                                                'found_by_crew': None,
                                                'requires_shutdown': False,
                                                })

        return

    def find_leaks (self):
        '''
        Loop over all your methods in the simulation and ask them to find some leaks.
        '''

        for m in self.state['methods']:
            m.find_leaks ()

        if self.state['t'].current_date.weekday() == 0:
            self.state['operator'].work_a_day()
            
        return

    def repair_leaks (self):
        '''
        Repair tagged leaks and remove from tag pool.
        '''
        for tag in self.state['tags']:
            if (self.state['t'].current_date - tag['date_found']).days  == self.parameters['repair_delay']:
                tag['status'] = 'repaired'
                tag['date_repaired'] = self.state['t'].current_date
                tag['repair_delay'] = (tag['date_repaired'] - tag['date_found']).days
        
        self.state['tags'] = [tag for tag in self.state['tags'] if tag['status'] == 'active']

        return

    def report (self):
        '''
        Daily reporting of leaks, repairs, and emissions.
        '''

        # Update timeseries
        self.timeseries['new_leaks'].append(sum(d['n_new_leaks'] for d in self.state['sites']))
        self.timeseries['cum_repaired_leaks'].append(sum(d['status'] == 'repaired' for d in self.state['leaks']))
        self.timeseries['daily_emissions_kg'].append(sum(d['rate'] for d in self.active_leaks))
        self.timeseries['n_tags'].append(len(self.state['tags']))

        print ('Day ' + str(self.state['t'].current_timestep) + ' complete!')
        
        return


    def finalize (self):
        '''
        Compile and write output files.
        '''
        print ('Finalizing simulation...')
        output_directory = os.path.join(self.parameters['working_directory'], self.parameters['output_folder'])
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)       
         
        # Attribute individual leak emissions to site totals
        for leak in self.state['leaks']:
            tot_emissions_kg = leak['days_active']*leak['rate']
            for site in self.state['sites']:
                if site['facility_ID'] == leak['facility_ID']:
                    site['total_emissions_kg'] += tot_emissions_kg
                    if leak['status'] == 'active':
                        site['active_leaks'] += 1
                    elif leak['status'] == 'repaired':
                        site['repaired_leaks'] += 1
                    break

        # Make maps and append site-level DD and MCB data
        for m in self.state['methods']:
            m.make_maps()
            m.site_reports()
    
        # Generate some dataframes           
        for site in self.state['sites']:
            del site['n_new_leaks']

        leak_df = pd.DataFrame(self.state['leaks'])
        time_df = pd.DataFrame(self.timeseries)
        site_df = pd.DataFrame(self.state['sites'])
 
        # Create some new variables for plotting
        site_df['cum_frac_sites'] = list(site_df.index)
        site_df['cum_frac_sites'] = site_df['cum_frac_sites']/max(site_df['cum_frac_sites'])
        site_df['cum_frac_emissions'] = np.cumsum(sorted(site_df['total_emissions_kg'], reverse = True))
        site_df['cum_frac_emissions'] = site_df['cum_frac_emissions']/max(site_df['cum_frac_emissions'])       
        
        leaks_active = leak_df[leak_df.status == 'active'].sort_values('rate', ascending = False)
        leaks_repaired = leak_df[leak_df.status == 'repaired'].sort_values('rate', ascending = False)
        
        leaks_active['cum_frac_leaks'] = list(np.linspace(0, 1, len(leaks_active)))
        leaks_active['cum_rate'] = np.cumsum(leaks_active['rate'])
        leaks_active['cum_frac_rate'] = leaks_active['cum_rate']/max(leaks_active['cum_rate'])
        
        if len(leaks_repaired) > 0:
            leaks_repaired['cum_frac_leaks'] = list(np.linspace(0, 1, len(leaks_repaired)))
            leaks_repaired['cum_rate'] = np.cumsum(leaks_repaired['rate'])
            leaks_repaired['cum_frac_rate'] = leaks_repaired['cum_rate']/max(leaks_repaired['cum_rate'])
    
        leak_df = leaks_active.append(leaks_repaired)
        
        # Write csv files
        leak_df.to_csv(output_directory + '/leaks_output_' + self.parameters['simulation'] + '.csv', index = False)
        time_df.to_csv(output_directory + '/timeseries_output_' + self.parameters['simulation'] + '.csv', index = False)
        site_df.to_csv(output_directory + '/sites_output_' + self.parameters['simulation'] + '.csv', index = False)
        
        # Make plots
        make_plots(leak_df, time_df, site_df, self.parameters['simulation'], output_directory)

        # Write metadata
        metadata = open(output_directory + '/metadata_' + self.parameters['simulation'] + '.txt','w')
        metadata.write(str(self.parameters) + '\n' +
        str(datetime.datetime.now()))
        metadata.close()
        
        # Return to original working directory
        os.chdir(self.parameters['working_directory'])

        print ('Results have been written to output folder.')
        print ('Simulation complete. Thank you for using the LDAR Simulator.')
        return


