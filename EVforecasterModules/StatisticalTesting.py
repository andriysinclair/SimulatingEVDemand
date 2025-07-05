# STATISTICAL TESTING

# 1 Run the following N times

# 11. Generate charging schedule
# 12. Transform to wide df
# 13. Gather 5-min demands for the week - save the array.

# 2 Collect 5-min demand vector from the pilot study

# 3 Collect N R^2 values comparing simulations and Pilot

# 4 Repeat using seasonal simulations and aggregate simulations

# 5 Perform statistical tests to hopefully show that seasonal simulations are better than aggregate.

# DONE

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config as cfg
import logging
import time
import pickle
from EVforecasterModules.charging_logic import charging_logic
from EVforecasterModules.demand_curves import output_full_long_df, output_wide_df, create_labels
from scipy.stats import mannwhitneyu

def simulate(N_sims, 
             df, 
             home_shift, 
             week, 
             battery_size_phev,
            battery_size_bev,
            car_types,
            charging_rates,
            home_charger_likelihood,
            work_charger_likelihood,
            public_charger_likelihood,
            min_stop_time_to_charge,
            SOC_charging_prob,
            test_index = None):

    slots_5_week = int(1440 * 7 / 5)  # 2016 5-min bins in a week

    results_matrix = np.zeros((N_sims, slots_5_week))
    sim_times = np.zeros(N_sims)

    for n in range(N_sims):

        sim_start_time = time.time()

        charging_df = charging_logic(df, 
                                     home_shift=home_shift, 
                                     travel_weeks=week, 
                                     test_index=test_index,
                                     battery_size_bev=battery_size_bev, 
                                     battery_size_phev=battery_size_phev, 
                                     car_types=car_types,
                                     charging_rates=charging_rates, 
                                     home_charger_likelihood=home_charger_likelihood,
                                     work_charger_likelihood=work_charger_likelihood, 
                                     public_charger_likelihood=public_charger_likelihood,
                                     min_stop_time_to_charge=min_stop_time_to_charge,
                                     SOC_charging_prob=SOC_charging_prob)
        
        charging_df = output_full_long_df(charging_df)
        wide_df = output_wide_df(charging_df)

        num_i = len(wide_df)

        demand_vector = wide_df.iloc[:, :-1].sum() # Total demand vector

        results_matrix[n, : ] = demand_vector.values / num_i   # Average Demand vector

        sim_times[n] = time.time() - sim_start_time

        print(f"Completed sim {n+1}/{N_sims} for weeks {week[0]}–{week[-1]} in {sim_times[n]:.2f}s", end="\r")



    return results_matrix, sim_times

def apply_ECA_overlay():
    pass


def return_R2(results_matrix, ECA_data):
        # Results for an RSS for each time slot for each simulation
        RSS = np.sum((results_matrix - ECA_data) ** 2, axis=1)

        logging.info(f"RSS shape: {RSS.shape}")

        TSS = np.sum((ECA_data - np.mean(ECA_data, keepdims=True)) ** 2)

        logging.info(f"TSS shape: {TSS.shape}")

        R_2 = 1 - RSS / TSS
        logging.info(f"R² (first 15): {R_2[:15]}")

        logging.info(f"R_2 shape (weeks, simulations): {R_2.shape}")

        return R_2

def plot_demand(results_matrix, x, x_labels, ECA_data=None):

    # Plot simulation vs ECA + R² distribution only for overall plot
    mean_curve = np.mean(results_matrix, axis=0)
    lower_bound = np.percentile(results_matrix, 2.5, axis=0)
    upper_bound = np.percentile(results_matrix, 97.5, axis=0)


    plt.figure(figsize=(15, 6))

    # Demand curve - plot just for simulation
    plt.subplot(2, 1, 1)
    plt.plot(x, mean_curve, label="Simulation Mean", color="blue")
    plt.fill_between(x, lower_bound, upper_bound, color='lightblue', alpha=0.7, label='95% CI')
    plt.ylabel('Demand (kWh/5min)/EV')
    plt.xticks(ticks=range(0, len(x_labels), 72), labels=x_labels[::72], rotation=45)
    plt.grid()

    if ECA_data is not None:
        plt.plot(x, ECA_data, label="Electric Chargepoint Analysis 2017", linestyle="--", color="orange")
    
    
    plt.tight_layout()
    plt.legend()


    '''
    ### MOVE To DIFFERENT FUNCTION ###

    # R² histogram
    plt.subplot(2, 1, 2)

    '''

def plot_R2(results_matrix, ECA_data):

    R_2 = return_R2(results_matrix=results_matrix, ECA_data=ECA_data)

    sns.histplot(R_2, kde=True, bins=10, color='steelblue')
    plt.xlabel('R²')
    plt.ylabel('Density')
    plt.grid()
    plt.tight_layout()

    return R_2



def violin_plot(Model1, Model2):
    pass

    

def surface_plot_3d():
    pass





    



def obtain_algo_perfromance(results_folder, n_min, n_step, n_max, N_sims = 10):

    num_steps = ((n_max - n_min) // n_step) + 1

    mean_sim_times = np.zeros(num_steps)

    sample_sizes = np.zeros(num_steps)

    for i, sample_size in enumerate(range(n_min, n_max+1, n_step)):

        logging.info(f"Running test for n_individuals = {sample_size} out of {n_max}")

        mean_sim_time = obtain_results(results_folder=results_folder, N_sims=N_sims, test_index=sample_size, testing_performance=True, simulate=True)

        mean_sim_times[i] = mean_sim_time

        sample_sizes[i] = sample_size

        logging.info(f"test {i+1} out of {num_steps}\n")

    # Calculating O(n)

    log_n = np.log10(sample_sizes)
    log_t = np.log10(mean_sim_times)

    slope, intercept = np.polyfit(log_n, log_t, deg=1)

    print(f"\nEstimated time complexity: O(n^{slope:.2f})")

    # Plotting results

    plt.plot(sample_sizes, mean_sim_times)

    plt.xlabel("Number of Individuals (n)")
    plt.ylabel("Mean Simulation Time over 10 simulations (seconds)")

    plt.grid()

    plot_path = plots_folder + f"SimulationPerformance_{n_min}_{n_step}_{n_max}.pdf"

    plt.savefig(plot_path, format="pdf")





if __name__ == "__main__":

    import matplotlib

    matplotlib.use("Agg")

    results_folder = cfg.root_folder + "/results"
    plots_folder = cfg.root_folder + "/plots/"

    # Loading in travel survey df

    travel_survey_path_aug = cfg.root_folder + "/dataframes/Ready_to_model_df_[2012, 2013, 2014, 2015, 2016, 2017, 2018].pkl"
    travel_survey_path = cfg.root_folder + "/dataframes/Ready_to_model_df_[2017].pkl"

    travel_survey_df = pd.read_pickle(travel_survey_path)
    travel_survey_df_aug = pd.read_pickle(travel_survey_path_aug)

    # Set up basic configuration for logging
    logging.basicConfig(level=logging.INFO)

    # Simulating for all weeks of the year for different values of home shift

    '''
    home_shifts = [0, 60]

    for hs in home_shifts:

        obtain_results(100, home_shift=hs, results_folder=results_folder, plots_folder=plots_folder, simulate=False)
    '''

    obtain_results(100, travel_survey_df=travel_survey_df, home_shift=60, 
                   results_folder=results_folder, plots_folder=plots_folder, simulate=False)
    
    obtain_results(100, travel_survey_df=travel_survey_df_aug, home_shift=60, 
                   results_folder=results_folder, plots_folder=plots_folder, simulate=False, suffix="aug")

    run_statistical_tests(home_shift_1=60, home_shift_2=60, suffix1="", suffix2="aug", plots_folder=plots_folder)
    #obtain_algo_perfromance(results_folder=results_folder, n_min=50, n_step=50, n_max=4000)

    
    # Simulating only for the weeks relevant to travel

    #obtain_results(100, travel_weeks_sim=list(range(39,53)),
    #                results_folder=results_folder, plots_folder=plots_folder, simulate=True)
    
    ####

    '''
    obtain_results(3, travel_weeks_sim=list(range(49,54)),
                   travel_weeks_eca=list(range(49,53)),
                results_folder=results_folder, plots_folder=plots_folder, simulate=True)

    '''