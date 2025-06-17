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
from charging_logic import charging_logic
from demand_curves import output_full_long_df, output_wide_df, create_labels

# Loading in travel survey df

travel_survey_path = cfg.root_folder + "/dataframes/Ready_to_model_df_[2017].pkl"
travel_survey_df = pd.read_pickle(travel_survey_path)


def return_R2():
    




def obtain_results(N_sims, results_folder, plots_folder=None,
                   start_week = 39,
                   travel_weeks_eca=list(range(39, 53)),
                   simulate=True,
                   test_index=None,
                   testing_performance=False,
                   home_shift = cfg.home_shift):
    
    # Load all weeks of ECA data individually
    
    ECA_data = np.zeros( (15, 2016) )

    for i, week in enumerate(range(39, 52  + 1)):
        # Load ECA target vector foe every week
        with open(results_folder + f'/y_ECA_{week}-{week}.pkl', 'rb') as f:
            y_eca = pickle.load(f)
            y_eca = np.array(y_eca)
            logging.debug(f"{week}, {y_eca.shape}")

            ECA_data[i, :] = y_eca

    with open(results_folder + f'/y_ECA_39-52.pkl', 'rb') as f:
        y_eca = pickle.load(f)
        y_eca = np.array(y_eca)
        logging.debug(y_eca.shape)

        ECA_data[-1, :] = y_eca

    # Stacking across sim dimension to make for easier calculation
    ECA_data = np.stack([ECA_data]*N_sims, axis=2)

    logging.debug(ECA_data.shape)


    # Run simulation
    if simulate:
        slots_5_week = int(1440 * 7 / 5)

        results_matrix = np.zeros((52 - 39 + 2, slots_5_week, N_sims))
        sim_times = np.zeros(N_sims)

        for n in range(N_sims):
            sim_start_time = time.time()

            for i, week in enumerate(range(39, 52+1)):

                charging_df = charging_logic(travel_survey_df, travel_weeks=[week], test_index=test_index)
                charging_df = output_full_long_df(charging_df)
                wide_df = output_wide_df(charging_df)

                num_i = len(wide_df)
                demand_vector = wide_df.iloc[:, :-1].sum()
                results_matrix[i, : , n] = demand_vector.values / num_i

                logging.info(f"Week {week} for sim {n+1} complete!")

            # Now do an overall simulation over all weeks

            charging_df = charging_logic(travel_survey_df, travel_weeks=list(range(1,52+1)), test_index=test_index)
            charging_df = output_full_long_df(charging_df)
            wide_df = output_wide_df(charging_df)

            num_i = len(wide_df)
            demand_vector = wide_df.iloc[:, :-1].sum()
            results_matrix[-1, : , n] = demand_vector.values / num_i

            logging.info(f"Overall weeks 1 -52 simulation complete")

            sim_times[n] = time.time() - sim_start_time
            logging.info(f"Simulation {n + 1} of {N_sims} complete in {sim_times[n]:.2f}s")

        # If testing performance, return immediately with timing info
        if testing_performance:
            return np.mean(sim_times)

        # Save output from last simulation
        x = demand_vector.index
        x_labels = create_labels(wide_df)

        with open(results_folder + f'/results_matrix-{N_sims}_3D.pkl', 'wb') as f:
            pickle.dump(results_matrix, f)
        with open(results_folder + f'/x.pkl', 'wb') as f:
            pickle.dump(x, f)
        with open(results_folder + f'/x_labels.pkl', 'wb') as f:
            pickle.dump(x_labels, f)

    # Load from file if not simulating
    else:
        with open(results_folder + f'/results_matrix-{N_sims}_3D.pkl', 'rb') as f:
            results_matrix = pickle.load(f)
        with open(results_folder + f'/x.pkl', 'rb') as f:
            x = pickle.load(f)
        with open(results_folder + f'/x_labels.pkl', 'rb') as f:
            x_labels = pickle.load(f)

        logging.debug(results_matrix.shape)

    # Guard R² and plotting fully under not testing_performance
    if not testing_performance:
        # R² Calculation

        # Results for an RSS for each time slot for each simulation
        RSS = np.sum((results_matrix - ECA_data) ** 2, axis=1)

        logging.debug(f"RSS shape: {RSS.shape}")


        TSS = np.sum((ECA_data - np.mean(ECA_data, axis=1, keepdims=True)) ** 2, axis=1)

        logging.debug(f"TSS shape: {TSS.shape}")

        R_2 = 1 - RSS / TSS
        logging.debug(f"R² (first 15): {R_2[:15]}")

        # Save R² values
        with open(results_folder + f'/R_2_{N_sims}_3D.pkl', 'wb') as f:
            pickle.dump(R_2, f)

        # Plot simulation vs ECA + R² distribution only for overall plot
        mean_curve = np.mean(results_matrix, axis=2)[-1,:]
        lower_bound = np.percentile(results_matrix, 2.5, axis=2)[-1,:]
        upper_bound = np.percentile(results_matrix, 97.5, axis=2)[-1,:]

        plt.figure(figsize=(15, 6))

        # Demand curve
        plt.subplot(2, 1, 1)
        plt.plot(x, mean_curve, label="Simulation Mean", color="blue")
        plt.plot(x, y_eca, label="Electric Chargepoint Analysis 2017", linestyle="--", color="orange")
        plt.fill_between(x, lower_bound, upper_bound, color='lightblue', alpha=0.7, label='95% CI')
        plt.ylabel('Demand (kWh/5min)/EV')
        plt.xticks(ticks=range(0, len(x_labels), 72), labels=x_labels[::72], rotation=45)
        plt.grid()
        plt.legend()

        # R² histogram
        plt.subplot(2, 1, 2)
        sns.histplot(R_2[-1,:], kde=True, bins=10, color='steelblue')
        plt.xlabel('R²')
        plt.ylabel('Density')
        plt.grid()

        plt.tight_layout()
        plot_path = plots_folder + f"sim_plot_{N_sims}_3D_homeshift{home_shift}.pdf"
        logging.info(f"Saved plot to {plot_path}")
        plt.savefig(plot_path, format="pdf")


    return results_matrix



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

    # Set up basic configuration for logging
    logging.basicConfig(level=logging.INFO)

    # Simulating for all weeks of the year

    obtain_results(3, results_folder=results_folder, plots_folder=plots_folder, simulate=False)


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