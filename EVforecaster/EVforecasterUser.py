import sys
import os
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# Add project root to sys.path
#sys.path.append(str(Path(__file__).resolve().parents[1]))

from EVforecasterModules import config as cfg
from EVforecasterModules.data_loaders import data_loader_end_to_end
from EVforecasterModules.charging_logic import charging_logic
from EVforecasterModules.StatisticalTesting import simulate, plot_demand, plot_R2
from EVforecasterModules.pilots import demand_curves_ECA

print(cfg.root_folder)

# Reset root logger handlers if already set (optional cleanup)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO)


class EVforecaster:

    def inspect_config_params(self):
        """
        inspect_config_params 

        Prints key parameters from config.py
        """        

        print("Please inspect parameters in EVforecaster/Modules/config.py\n")

        key_parameters = ["battery_size_bev", "SOC_charging_prob", "battery_size_phev",
                          "car_types", "charger_likelihood", "charging_rates", "home_shift", "min_stop_time_to_charge",
                          "root_folder"]

        for name in dir(cfg):
            if name in key_parameters:
                attr = getattr(cfg, name)
                print(f"{name}")
                print(f"{attr}")
                print("")

        print(f"For more information please refer to EVforecaster/Modules/config.py")


    def __init__(self, 
                 travel_years,
                 data_folder= cfg.root_folder + "/data/",
                 plots_folder = cfg.root_folder + "/plots/",
                 results_folder = cfg.root_folder + "/results/",
                 survey_years = list(range(2012,2019))):
        """
        __init__ 

        Initialises EVforecaster class

        Args:
            travel_years (list): List of travel years to use for simulation, if using a single year please input a list with a single entry
            data_folder (str, optional): data folder path. Defaults to cfg.root_folder+"/data/".
            plots_folder (str, optional): plots folder path. Defaults to cfg.root_folder+"/plots/".
            survey_years (list, optional): Survey years on which to filter the NTS. Defaults to list(range(2012,2019)).

        Raises:
            FileNotFoundError: If the NTS-UK datasets do not exist in the /data folder
        """        
        
        # This is the dataset merged and filtered on travel years
        self.dataset_name = f"{travel_years[0]}-{travel_years[-1]}-merged.pkl"
        self.data_folder = data_folder
        self.results_folder = results_folder
        self.trip_path = None
        self.day_path = None
        self.household_path = None
        self.plots_folder = plots_folder
        self.df = None
        self.survey_years = survey_years
        self.travel_years = travel_years

        # Key files from NTS-UK

        file_names = ["trip_eul_2002-2023.tab", "day_eul_2002-2023.tab", "household_eul_2002-2023.tab"]

        # Checking if files exist in the data folder

        for filename in file_names:
            filepath = os.path.join(data_folder, filename)

            if not os.path.isfile(filepath):
                raise FileNotFoundError(
                    f"\nMissing required file: {filename}\n"
                    "Please download the UK-NTS dataset from:\n"
                    "  https://doi.org/10.5255/UKDA-SN-5340-14\n"
                    f"Then move the required files into: {data_folder}"
                )
            
        # Appending paths to instance attributes (IF FOUND IN DIR)

        self.trip_path = data_folder + file_names[0]
        logging.debug(f"Trip path: {self.trip_path}")

        self.day_path = data_folder + file_names[1]
        logging.debug(f"Day path: {self.day_path}")

        self.household_path = data_folder + file_names[2]
        logging.debug(f"Household path: {self.household_path}")

        print("All required files are present.")

        # Inspect the parameters in config.py

        self.inspect_config_params()

        # Checking if dataset given by dataset name

        filepath = os.path.join(data_folder, self.dataset_name)

        logging.debug(f"Filepath to merged dataset: {filepath}")

        if not os.path.isfile(filepath):
            print(f"No merged dataset found")
            print("Ceating Dataset...")

            self.df = data_loader_end_to_end(travel_year=self.travel_years,
                                             trip_path=self.trip_path,
                                             day_path=self.day_path,
                                             household_path=self.household_path,
                                             survey_years=self.survey_years,
                                             trip_cols_to_keep=cfg.trip_cols_to_keep,
                                             trip_purpouse_mapping=cfg.trip_purpouse_mapping,
                                             trip_type_mapping=cfg.trip_type_mapping,
                                             day_cols_to_keep=cfg.day_cols_to_keep,
                                             household_cols_to_keep=cfg.household_cols_to_keep)
            
            self.df.to_pickle(filepath)

            print(f"Successfully exported dataset to {filepath}")

        else:
            print(f"Merged dataset found. Ready to forecast!")

            self.df = pd.read_pickle(filepath)

        logging.info(f"Travel years in dataset: {self.df["TravelYear"].unique()}")


    def generate_forecasts(self, N_sims, weeks, home_shift, experiment_name,
                                results_folder = cfg.root_folder + "/results/",
                                battery_size_bev= cfg.battery_size_bev,
                                battery_size_phev= cfg.battery_size_phev,
                                car_types= cfg.car_types,
                                charging_rates= cfg.charging_rates,
                                home_charger_likelihood= cfg.charger_likelihood["home"],
                                work_charger_likelihood= cfg.charger_likelihood["work"],
                                public_charger_likelihood= cfg.charger_likelihood["other"],
                                min_stop_time_to_charge = cfg.min_stop_time_to_charge,
                                SOC_charging_prob = cfg.SOC_charging_prob,
                                ECA_overlay = None):
        
        results_dict = {}  

        df = self.df.copy()

        # Loading labels for plotting

        with open(self.results_folder + f'/x.pkl', 'rb') as f:
            x = pickle.load(f)
        with open(self.results_folder + f'/x_labels.pkl', 'rb') as f:
            x_labels = pickle.load(f)

        # Creating paths for plots and results
        array_path = self.results_folder + experiment_name + ".pkl"
        plot_path = self.plots_folder + experiment_name + ".pdf"

        # Check if simulation already exists

        if not os.path.isfile(array_path):
        
            results_matrix, sim_times = simulate(N_sims=N_sims,
                                                home_shift=home_shift,
                                                week=weeks,
                                                df=df,
                                                battery_size_bev=battery_size_bev,
                                                battery_size_phev=battery_size_phev,
                                                car_types=car_types,
                                                charging_rates=charging_rates,
                                                home_charger_likelihood=home_charger_likelihood,
                                                work_charger_likelihood=work_charger_likelihood,
                                                public_charger_likelihood=public_charger_likelihood,
                                                min_stop_time_to_charge=min_stop_time_to_charge,
                                                SOC_charging_prob=SOC_charging_prob)
            
        if os.path.isfile(array_path):

            logging.info(f"Simulation with name: {experiment_name} already exists in /results. Loading ...")
            
            # Load the data from pickle

            with open(array_path, "rb") as f:
                results_dict = pickle.load(f)

            results_matrix = results_dict["results_matrix"]
            sim_times = results_dict["sim_times"]
        
        # Moving to results folder

        logging.info(f"Final post-simulation results matrix shape: {results_matrix.shape}")



        # If no ECA overlay
        if ECA_overlay is None:

            logging.info(f"ECA not required")

            logging.info(f"Generating plot and saving to: {plot_path}")

            plot_demand(results_matrix=results_matrix, x=x, x_labels=x_labels)

            plt.savefig(plot_path, format="pdf")

        # Add ECA overlay if desired

        if ECA_overlay is not None:
            logging.info(f"Applying ECA overlay")

            # Ensure that the ECA file is in data
            ECA_name = "electric-chargepoint-analysis-2017-raw-domestics-data.csv"

            ECA_path = os.path.join(self.data_folder, ECA_name)

            if os.path.isfile(ECA_path):
                print(f"{ECA_name} successfully found in /data directory. Proceeding...")

                print(f"Checking for previously created ECA overlay for weeks: {ECA_overlay[0]} - {ECA_overlay[-1]}")

                # Check if ECA overlay file exists

                ECA_demand_name = f"y_ECA_{ECA_overlay[0]}-{ECA_overlay[-1]}.pkl"

                ECA_demand_path = os.path.join(self.results_folder, ECA_demand_name)

                if not os.path.isfile(ECA_demand_path):
                    logging.info(f"ECA demand curves, {ECA_demand_name}, do not exist in /results. Creating...")

                    y_ECA = demand_curves_ECA(ECA_df_path=ECA_path,
                                              results_folder=self.results_folder,
                                              ECA_weeks=ECA_overlay,
                                              )
                    
                    logging.info("ECA demand curve created!")

                    # Saving to pickle

                    y_ECA.to_pickle(ECA_demand_path)

                if os.path.isfile(ECA_demand_path):
                    logging.info(f"Successfully found {ECA_demand_name} in /results")

                    y_ECA = pd.read_pickle(ECA_demand_path)


                ## HERE APPLY THE ECA overlay

                logging.info(f"Y_eca shape: {y_ECA.shape}")

                logging.debug(f"ECA shape: {y_ECA.shape}. ECA type: {type(y_ECA)}")

                logging.info(f"Generating plot and saving to: {plot_path}. ECA required")

                plt.subplot(2,1,1)

                plot_demand(results_matrix=results_matrix, x=x, x_labels=x_labels, ECA_data=y_ECA)

                ### HERE WE MUST GENERATE R_2 also + comparison tables to CSV??

                # Re-shaping ECA data

                y_ECA = y_ECA.reshape(1, -1)

                logging.info(f"Y_eca shape: {y_ECA.shape}")

                plt.subplot(2,1,2)

                R_2 = plot_R2(results_matrix=results_matrix, ECA_data=y_ECA)

                plt.savefig(plot_path, format="pdf")

                # Saving R_2 to results
                results_dict["R2"] = R_2

            if not os.path.isfile(ECA_path):
                raise FileNotFoundError(f"{ECA_name} not found in /data directory.\n Please download Electric Chargepoint Analysis data \n from 'https://www.gov.uk/government/statistics/electric-chargepoint-analysis-2017-domestics' \n and move inside /data")

        ### SAVING RESULTS

        logging.info(f"saving results to {array_path}")

        results_dict["results_matrix"] = results_matrix
        results_dict["sim_times"] = sim_times

        with open(array_path, "wb") as f:
            pickle.dump(results_dict, f)

    
    def run_experiment(experiment1_r2_path, experiment2_r2_path):

        with open(experiment1_r2_path, "rb") as f:
            results_dict1 = pickle.load(f)

        with open(experiment2_r2_path, "rb") as f:
            results_dict2 = pickle.load(f)

        # Extract R2 from experiment 1

        R2_1 =  results_dict1["R2"]
        R2_2 =  results_dict2["R2"]

        #  Run experiments

        stat, p = mannwhitneyu(R2_1, R2_2, alternative='two-sided')
        logging.info(f"Mann–Whitney U test statistic: {stat:.4f}, p-value: {p:.4f}")


