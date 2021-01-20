# ------------------------------------------------------------------------------
# Program:     The LDAR Simulator (LDAR-Sim) 
# File:        LDAR-Sim main
# Purpose:     Interface for parameterizing and running LDAR-Sim.
#
# Copyright (C) 2018-2020  Thomas Fox, Mozhou Gao, Thomas Barchyn, Chris Hugenholtz
#    
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, version 3.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# ------------------------------------------------------------------------------

from batch_reporting import *
from ldar_sim_run import *
import os
import datetime
import warnings
import multiprocessing as mp
import boto3 # for downloading data from AWS

if __name__ == '__main__':
    # ------------------------------------------------------------------------------
    # -----------------------------Global parameters--------------------------------
    wd = "../inputs_template/"
    wd = os.path.abspath (wd) + "/"
    program_list = ['P_ref','P_alt', 'P_alt2', 'P_cont']  # Programs to compare; Position one should be the reference program (P_ref)
    n_processes = None  # Number of processes to use, None = all, 1 = one virtual core, and so on.
    print_from_simulations = True  # Print informational messages from within the simulations
    warnings.filterwarnings('ignore')    # Temporarily mute warnings
    
    #-------------------------------------------------------------------------------
    #------------------------------Check ERA5 data in the working directory---------
    def check_ERA5_file(Dir,era_file): 
        ncfiles = [] 
        for file in os.listdir(Dir):
            if file.endswith(".nc"):
                ncfiles.append(file)
        if wfile in ncfiles: 
            print ("Weather data checked")
        else:
            print ("I will donwload data for you...")
            access_key = "AKIAIQ3KWS7VA4KSJIKA" # the access key and secret key will change in the future, these two keys are currently corresponding to my AWS account 
            secret_key = "5b6a8z2lLMNkRdsEceL/gwYRNh2Ar5sPJjFwDD7n"
            s3 = boto3.client('s3', aws_access_key_id=access_key , aws_secret_access_key=secret_key)
            s3.download_file('eratest',era_file,r'{}/ERA5_AB_1x1_hourly_2015_2019.nc'.format(Dir))
            print ("Weather data downloaded")
            
    era_file = r"ERA5_AB_1x1_hourly_2015_2019.nc"        
    check_ERA5_file(wd,era_file)
    # -----------------------------Set up programs----------------------------------
    programs = []
    for p in range(len(program_list)):
        file = wd + program_list[p] + '.txt'
        exec(open(file).read())
        programs.append(eval(program_list[p]))

    n_simulations = programs[0]['n_simulations']
    spin_up = programs[0]['spin_up']
    ref_program = program_list[0]
    write_data = programs[0]['write_data']
    output_directory = wd + 'outputs/'

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Set up simulation parameter files
    simulations = []
    for i in range(n_simulations):
        for j in range(len(programs)):
            opening_message = 'Simulating program ' + str(j + 1) + ' of ' + str(len(programs)) + '; simulation ' + \
                                str(i + 1) + ' of ' + str(n_simulations)
            simulations.append([{'i': i, 'program': programs[j], 'wd': wd, 'opening_message': opening_message,
                                'print_from_simulation': print_from_simulations}])

    # Perform simulations in parallel
    with mp.Pool(processes=n_processes) as p:
        res = p.starmap(ldar_sim_run, simulations)

    # Do batch reporting
    if write_data:
        # Create a data object...
        reporting_data = BatchReporting(output_directory, programs[0]['start_year'], spin_up, ref_program)
        if n_simulations > 1:
            reporting_data.program_report()
            if len(programs) > 1:
                reporting_data.batch_report()
                reporting_data.batch_plots()

    # Write metadata
    metadata = open(output_directory + '/metadata.txt', 'w')
    metadata.write(str(programs) + '\n' +
                   str(datetime.datetime.now()))

    metadata.close()

    # Write sensitivity analysis data on a program by program basis
    sa_df = pd.DataFrame(res)
    if 'program' in sa_df.columns:
        for program in sa_df['program'].unique():
            sa_out = sa_df.loc[sa_df['program'] == program, :]
            sa_outfile_name = os.path.join(wd, 'sensitivity_analysis', 'sensitivity_' + program + '.csv')
            sa_out.to_csv(sa_outfile_name, index=False)

