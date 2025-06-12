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
import config as cfg
import logging
import time
import pickle
from charging_logic import charging_logic
from demand_curves import output_full_long_df, output_wide_df, create_labels

# Loading in travel survey df

travel_survey_path = cfg.root_folder + "/dataframes/Ready_to_model_df_[2017].pkl"
travel_survey_df = pd.read_pickle(travel_survey_path)


def obtain_results(sim_times, results_folder, plots_folder, simulate=True):

    # Loading results from pilot
    with open(results_folder + f'/y_ECA.pkl', 'rb') as f:
        y_eca = pickle.load(f)

    if simulate:

        slots_5_week = int(1440*7/5)

        results_matrix = np.zeros( (sim_times, slots_5_week))

        for n in range(sim_times):

            sim_start_time = time.time()

            ### LOGIC ####

            # Obtain charging schedule df
            charging_df = charging_logic(travel_survey_df)

            # Add extra bits needed before wide transformation
            charging_df = output_full_long_df(charging_df)

            # Perform wide transformation
            wide_df = output_wide_df(charging_df)

            num_i  = len(wide_df)

            # Obtain the 5-min weekly demand vector
            demand_vector = wide_df.iloc[:,:-1].sum()

            # Store the results
            results_matrix[n, :] = demand_vector.values / num_i   # To get /EV demand curves

            ##############

            sim_end_time = time.time()

            sim_time = sim_end_time - sim_start_time

            logging.info(f"Simulation {n+1} out of {sim_times} complete")
            logging.info(f"Complete in {sim_time:.2f}s")

            # Takes around 23 seconds to run a single simulation

        # SIMULATION COMPLETE

        x = demand_vector.index

        x_labels = create_labels(wide_df)

        with open(results_folder + f'/results_matrix-{sim_times}.pkl', 'wb') as f:
            pickle.dump(results_matrix, f)

        with open(results_folder + f'/x.pkl', 'wb') as f:
            pickle.dump(x, f)

        with open(results_folder + f'/x_labels.pkl', 'wb') as f:
            pickle.dump(x_labels, f)

    if not simulate:

        with open(results_folder + f'/results_matrix-{sim_times}.pkl', 'rb') as f:
            results_matrix = pickle.load(f)

        with open(results_folder + f'/x.pkl', 'rb') as f:
            x = pickle.load(f)

        with open(results_folder + f'/x_labels.pkl', 'rb') as f:
            x_labels = pickle.load(f)

    # Now to collect R^2s


    RSS = np.sum((results_matrix - y_eca) ** 2, axis=1)
    TSS = np.sum((y_eca - np.mean(y_eca)) ** 2)
    R_2 = 1 - RSS / TSS


    logging.debug(f"R_2: {R_2}")

    # Saving R_2 to results

    with open(results_folder + f'/R_2_{sim_times}.pkl', 'wb') as f:
        pickle.dump(R_2, f)


    # Now to plot Simulation Results

    # Collect mean demand + CI
    mean_curve = np.mean(results_matrix, axis=0)
    lower_bound = np.percentile(results_matrix, 2.5, axis=0)
    upper_bound = np.percentile(results_matrix, 97.5, axis=0)

    plt.figure(figsize=(15,6))

    plt.subplot(1,2,1)
    
    plt.plot(x, mean_curve, label="Mean", color="blue")
    plt.fill_between(x, lower_bound, upper_bound, color='lightblue', alpha=0.4, label='95% CI')
    plt.ylabel('Demand (kWh/5 minutes)/ EV')
    #plt.title('Simulated Weekly EV Charging Demand')
    plt.xticks(ticks=range(0, len(x_labels), 72), labels=x_labels[::72], rotation=45)
    plt.grid()

    plt.legend()

    plt.tight_layout()

    # Now to plot pilot results alongside

    plt.subplot(1,2,2)

    plt.tight_layout()
    plt.plot(x, y_eca, label="Electric Chargepoint Analysis 2017", linestyle="--", color="orange")
    plt.plot(x, mean_curve, label="Simulation Mean", color="blue")
    #plt.ylabel('Demand (kWh/5 minutes)/ EV')
    #plt.title('Simulated Weekly EV Charging Demand')
    plt.xticks(ticks=range(0, len(x_labels), 72), labels=x_labels[::72], rotation=45)
    plt.grid()

    plt.legend()


    ## Saving Plot ##

    plot_path = plots_folder + f"sim_plot_{sim_times}.pdf"
    logging.info(plot_path)
    
    plt.savefig(plot_path, format="pdf")

    return results_matrix


if __name__ == "__main__":

    import matplotlib

    matplotlib.use("Agg")

    results_folder = cfg.root_folder + "/results"
    plots_folder = cfg.root_folder + "/plots/"

    # Set up basic configuration for logging
    logging.basicConfig(level=logging.INFO)

    obtain_results(3, results_folder=results_folder, plots_folder=plots_folder, simulate=False)

