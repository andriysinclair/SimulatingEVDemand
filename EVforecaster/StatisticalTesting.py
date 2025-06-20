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
from scipy.stats import mannwhitneyu


def return_R2(results_matrix, ECA_data, N_sims, home_shift, suffix=""):
        # Results for an RSS for each time slot for each simulation
        RSS = np.sum((results_matrix - ECA_data) ** 2, axis=1)

        logging.debug(f"RSS shape: {RSS.shape}")

        TSS = np.sum((ECA_data - np.mean(ECA_data, axis=1, keepdims=True)) ** 2, axis=1)

        logging.debug(f"TSS shape: {TSS.shape}")

        R_2 = 1 - RSS / TSS
        logging.debug(f"R² (first 15): {R_2[:15]}")

        # Save R² values
        with open(results_folder + f'/R_2_{N_sims}_3D_homeshift={home_shift}{suffix}.pkl', 'wb') as f:
            pickle.dump(R_2, f)

        logging.info(f"R_2 shape (weeks, simulations): {R_2.shape}")
        return R_2

def plot_demand_R2(results_matrix, ECA_data, x, x_labels, week, N_sims, home_shift, R_2, plots_folder, suffix=""):

    week_keys = list(range(39,53)) + ["overall"]
    matrix_index_values = list(range(0, 16))

    week_map = dict(zip(week_keys, matrix_index_values))

    logging.info(week_map)


    # Plot simulation vs ECA + R² distribution only for overall plot
    mean_curve = np.mean(results_matrix, axis=2)[week_map[week],:]
    lower_bound = np.percentile(results_matrix, 2.5, axis=2)[week_map[week],:]
    upper_bound = np.percentile(results_matrix, 97.5, axis=2)[week_map[week],:]

    plt.figure(figsize=(15, 6))

    # Demand curve
    plt.subplot(2, 1, 1)
    plt.plot(x, mean_curve, label="Simulation Mean", color="blue")
    plt.plot(x, ECA_data[week_map[week], :, 0], label="Electric Chargepoint Analysis 2017", linestyle="--", color="orange")
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
    plot_path = plots_folder + f"sim_plot_{N_sims}_3D_homeshift{home_shift}{suffix}.pdf"
    logging.info(f"Saved plot to {plot_path}")
    plt.savefig(plot_path, format="pdf")

def violin_plot(Model1, Model2):
    pass

    

def surface_plot_3d():
    pass



def obtain_results(N_sims, results_folder, home_shift, plots_folder, travel_survey_df,
                   simulate=True,
                   test_index=None,
                   testing_performance=False,
                   plot=True,
                   suffix=""):
    
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

                charging_df = charging_logic(travel_survey_df, home_shift=home_shift,  travel_weeks=[week], test_index=test_index)
                charging_df = output_full_long_df(charging_df)
                wide_df = output_wide_df(charging_df)

                num_i = len(wide_df)
                demand_vector = wide_df.iloc[:, :-1].sum()
                results_matrix[i, : , n] = demand_vector.values / num_i

                logging.info(f"Week {week} for sim {n+1} complete!")

            # Now do an overall simulation over all weeks

            charging_df = charging_logic(travel_survey_df, home_shift=home_shift, travel_weeks=list(range(1,52+1)), test_index=test_index)
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

        with open(results_folder + f'/results_matrix-{N_sims}_3D_homeshift={home_shift}{suffix}.pkl', 'wb') as f:
            pickle.dump(results_matrix, f)
        with open(results_folder + f'/x.pkl', 'wb') as f:
            pickle.dump(x, f)
        with open(results_folder + f'/x_labels.pkl', 'wb') as f:
            pickle.dump(x_labels, f)

    # Load from file if not simulating
    else:
        with open(results_folder + f'/results_matrix-{N_sims}_3D_homeshift={home_shift}{suffix}.pkl', 'rb') as f:
            results_matrix = pickle.load(f)
        with open(results_folder + f'/x.pkl', 'rb') as f:
            x = pickle.load(f)
        with open(results_folder + f'/x_labels.pkl', 'rb') as f:
            x_labels = pickle.load(f)

        logging.debug(results_matrix.shape)

    # Guard R² and plotting fully under not testing_performance
    if not testing_performance:

        # R² Calculation
        # By Week

        R_2 = return_R2(results_matrix=results_matrix, 
                        ECA_data=ECA_data, 
                        N_sims=N_sims,
                        home_shift=home_shift,
                        suffix=suffix)


        # R² Calculation
        # By overall vs Week

        # This is the final row of the results matrix
        overall_results = results_matrix[-1:,:,:]
        logging.debug(f"Overall results shape: {overall_results.shape}")

        # Now we need to stack it

        overall_results = np.repeat(overall_results, repeats=results_matrix.shape[0], axis=0)

        logging.debug(f"Overall results shape: {overall_results.shape}")

        logging.debug(results_matrix.shape)

        R_2_overall_v_week = return_R2(results_matrix=overall_results, 
                                       ECA_data=ECA_data,
                                        N_sims=N_sims,
                                        home_shift=home_shift,
                                        suffix=suffix)

        logging.debug(R_2_overall_v_week.shape)

        if plot:
            plot_demand_R2(results_matrix=results_matrix,
                        ECA_data=ECA_data,
                        plots_folder=plots_folder,
                        x=x,
                        x_labels=x_labels,
                        week="overall",
                        N_sims=N_sims,
                        home_shift=home_shift,
                        R_2=R_2,
                        suffix=suffix)
        

        # Calculating mean and variance of final R_2 stats to put in a nice latex and save to the correct path
        # And test statistic

        vs_week_mean = np.round(  np.mean(R_2, axis=1)  ,3)
        vs_week_sd = np.round(     np.std(R_2, axis=1)   ,3)

        vs_overall_mean = np.round(   np.mean(R_2_overall_v_week, axis=1)   ,3)
        vs_overall_sd = np.round(     np.std(R_2_overall_v_week, axis=1)    ,3)

        logging.info(f"Mean week-by-week: {vs_week_mean}")
        logging.info(f"Sd week-by-week: {vs_week_sd}")

        logging.info(f"Mean overall: {vs_overall_mean}")
        logging.info(f"Sd overall: {vs_overall_sd}")


    # Run Wilcoxon and t-test for differences


    return results_matrix, R_2, R_2_overall_v_week

def run_statistical_tests(home_shift_1, home_shift_2, suffix1, suffix2, plots_folder):
    # Running Man-Whitney test unpaired

    results_matrix0, R_2_0, R_2_overall_v_week0 = obtain_results(N_sims=100, home_shift=home_shift_1, 
                                                              results_folder=results_folder, 
                                                              plots_folder=plots_folder, simulate=False,
                                                              plot=False, suffix=suffix1,
                                                              travel_survey_df=travel_survey_df)
    
    results_matrix60, R_2_60, R_2_overall_v_week60 = obtain_results(N_sims=100, home_shift=home_shift_2, 
                                                            results_folder=results_folder, 
                                                            plots_folder=plots_folder, simulate=False,
                                                            plot=False, suffix=suffix2,
                                                            travel_survey_df=travel_survey_df)
    
    # Getting the overall R2 result only

    R_2_shift1 = R_2_0[-1,:]
    R_2_shift2 = R_2_60[-1,:]
    
    logging.info(f"shape R2s: {R_2_shift1.shape}, {R_2_shift2.shape}")

    # Running man Whitney test to test performance in shift

    stat, p = mannwhitneyu(R_2_shift1, R_2_shift2, alternative='two-sided')
    logging.info(f"Mann–Whitney U test statistic: {stat:.4f}, p-value: {p:.4f}")

    # Now testing if weekly forecast outperforms aggregate forcast

    logging.info(R_2_60.shape)

    df = pd.DataFrame()

    df_data = []



    for i in range(0,15):
        week = 39 + i if i < 14 else "Overall"

        # Append individual R²s for week-specific model
        for r2 in R_2_60[i, :]:
            df_data.append({
                "R2": r2,
                "Week": week,
                "Model": "Week-by-week"
            })

        # Append individual R²s for overall model
        for r2 in R_2_overall_v_week60[i, :]:
            df_data.append({
                "R2": r2,
                "Week": week,
                "Model": "Overall-by-week"
            })


        stat, p = mannwhitneyu(R_2_60[i,:], R_2_overall_v_week60[i,:], alternative='two-sided')
        logging.info(f"Mann–Whitney U test statistic for week {39+i}. week-by-week vs overall: {stat:.4f}, p-value: {p:.4f}")


    # Violin plots of week-by-week vs forecasting model

    # Build DataFrame
    df = pd.DataFrame(df_data)

    logging.info(df.head())

    plt.figure(figsize=(10,5))

    plt.grid()

    sns.boxplot(data=df, x="Week", y="R2", hue="Model", dodge=True)

    plot_path = plots_folder + "box_plot.pdf"
    logging.info(f"Saved plot to {plot_path}")
    plt.savefig(plot_path, format="pdf")
    



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