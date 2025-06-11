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


def obtain_results(sim_times, results_folder, simulate=True):

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

        return results_matrix

    if not simulate:

        with open(results_folder + f'/results_matrix-{sim_times}.pkl', 'rb') as f:
            results_matrix = pickle.load(f)

        with open(results_folder + f'/x.pkl', 'rb') as f:
            x = pickle.load(f)

        with open(results_folder + f'/x_labels.pkl', 'rb') as f:
            x_labels = pickle.load(f)

    # Now to plot ...

    # Collect mean demand

    mean_curve = np.mean(results_matrix, axis=0)

    plt.plot(x, mean_curve)
    


if __name__ == "__main__":

    # Set up basic configuration for logging
    logging.basicConfig(level=logging.INFO)

    obtain_results(3)

