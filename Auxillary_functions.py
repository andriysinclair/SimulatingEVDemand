import logging
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copulas.multivariate import GaussianMultivariate

# Configure basic logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

# Importing dataframe for experimentation

car_df = pd.read_pickle("/home/trapfishscott/Cambridge24.25/Energy_thesis/Data/df_car.pkl")

def filter_df(df, weekday, year):

    df = df.copy()

    if "TravelWeekDay_B03ID" not in df.columns:
        raise ValueError(f"'TravelWeekDay_B03ID' not in DataFrame columns. DataFrame columns: {df.columns}")

    if weekday == 1:
        df = df[df["TravelWeekDay_B03ID"]==1]

    elif weekday == 2:
        df = df[df["TravelWeekDay_B03ID"]==2]

    else:
        raise ValueError(f"Weekday is {weekday}. Must be int(1) for weekday or int(2) for weekend ")
    
    
    if not (2002 <= year <= 2023):  # Correct range check
        raise ValueError(f"Year is {year}. Must be between 2002 and 2023.")

    logging.debug(f"Filtering where year={year}")

    df = df[df["TravelYear"] == year]

    return df

def return_num_journey_prob(df, weekday, year, plots=True):

    df = df.copy()

    df = filter_df(df, weekday, year)
    
    x = df.groupby(["IndividualID", "TravDay"]).count()["JourSeq"]
    logging.debug(f"Counts of Individual Car Journeys by travel day: {x.head(10)}")

    if plots is True:
        logging.info("A histogram of number of journeys per person")
        plt.hist(x.values)
        plt.title("A histogram of number of journeys made.")
        plt.xlabel("Number of journeys")
        plt.show()
    

    logging.info("As we can see more than 10 car journeys a day are infrequent. So we shall remove these")

    cut_off_val = 10

    if cut_off_val > 19:
        raise ValueError(f"cut-off value is {cut_off_val}. Max possible value is 19. Recommended 10")

    x = x[x.values <= cut_off_val]

    logging.debug("Creating a matrix with first column individual ID and remaining columns corresponding to num of joourneys for each travel day")
    r = x.to_frame().reset_index()
    r = r.pivot_table(index="IndividualID", columns="TravDay", values="JourSeq", aggfunc="sum", fill_value=0)
    r = r.reset_index()

    travel_matrix = r.to_numpy()
    logging.debug(f"First 5 rows of travel matrix {travel_matrix[:5]}")
    logging.debug(f"Shape of travel matrix: {travel_matrix.shape}")

    # Returning mean for each row
    #trips_per_day = np.zeros((travel_matrix.shape[1], cut_off_val+1))
    for col in range(1,travel_matrix.shape[1]):
        unique, counts = np.unique(travel_matrix[:,col], return_counts=True)
        logging.debug(f"unique vals: {unique}")
        logging.debug(f"unique counts: {counts}")

    logging.info("We can see that across travel days the distribution of journey numbers by car is quite similar")
    
    logging.info("Obtaining frequency of trip counts over the full travel matrix")
    unique_values, counts = np.unique(travel_matrix[:,1:], return_counts=True)
    logging.debug(f"unique values: {unique_values}")
    logging.debug(f"unique counts: {counts}")
    sum_counts = np.sum(counts)
    logging.debug(f"sum of unique counts: {sum_counts}")

    logging.info("Returning probability vector")
    prob_vec = np.zeros(len(unique_values+1))
    for i in unique_values:
        prob_vec[i] = counts[i]/sum_counts

    logging.debug(f"Probability vector: {prob_vec}")
    logging.debug(f"Confirming p vector sums to 1: {np.sum(prob_vec)}")

    '''
    logging.debug("Making a dictionary of all unique individuals and number of car journeys in each travel day")
    logging.debug("Obtaining a list of all unique indiviudals")
    unique_i = np.unique(np.array(x.index.to_list())[:,0])
    logging.debug(f"{unique_i[:5]}")
    '''

    # Test Suites: Make sure unique values match, Make sure stuff matches at random
    logging.info(f"Returning probability vector of length {cut_off_val}, each entries is the probability of individual taking x trips on day t")
    return {"x": x, "travel_matrix": travel_matrix, "p_vec": prob_vec}

    #return individuals

#return_num_journey_prob(df=car_df, weekday=1, year=2023, plots=False)


def return_journey_seq(df, weekday, year):

    df = df.copy()

    df = filter_df(df,weekday, year)

    logging.info("Displaying propotions of different trip types")
    
    info = df["TripType"].value_counts(normalize=True).reset_index()

    #1 work
    #2 other
    #3 home

    mapping = {(2,3): "other->home",
            (3,2): "home->other",
            (3,1): "home->work",
            (2,2): "other->other",
            (1,3): "work->home",
            (1,2): "work->other",
            (2,1): "other->work"}

    info["TripType_mapped"] = info["TripType"].map(mapping)

    logging.info(info)
    
    logging.info("Calculating probabilities of different trip sequences for different trip lengths")
    logging.debug("Obtaining a trip type for every trip made by every individual on every travel day")
    g1 = car_df.groupby(["IndividualID", "TravDay", "JourSeq"])[["TripType"]].first()
    logging.debug(g1.head())
    logging.debug("Obtaining a list of journey types for each individual on each travel day")
    g1 = g1.groupby(["IndividualID", "TravDay"])[["TripType"]].agg(list)
    g1 = g1.reset_index()
    logging.debug(g1.head())

    logging.debug("Calculating number of trips for each travel day")
    g1["SumTrips"] = g1["TripType"].apply(lambda x: len(x))

    logging.debug(g1.head())

    cut_off_val = 10
    logging.info(f"Applying cut off at > {cut_off_val} trips")
    if cut_off_val > 19:
        raise ValueError(f"cut-off value is {cut_off_val}. Max possible value is 19. Recommended 10")

    g1 = g1[g1["SumTrips"] <= cut_off_val]

    logging.debug("For Travel Days with no travel leaving empty values")

    # Days 1,2..7 repeated unique individual number of times
    days = np.tile(np.arange(1,8), len(g1["IndividualID"].unique()))
    individuals = [i for i in g1["IndividualID"].unique() for _ in range(7)]
    new_df = pd.DataFrame({"IndividualID": individuals,
                        "TravDay": days })
    # Merging on idnividual and days and leaving NaNs for those individuals who did not travel on a given day
    merged_df = new_df.merge(g1, on=["IndividualID", "TravDay"], how="left")
    merged_df = merged_df.fillna(0)
    # Converting to strings for values counts
    merged_df["TripType"] = merged_df["TripType"].apply(lambda x: str(x))

    logging.debug(merged_df.iloc[5:10,:])

    logging.info(f"There are {len(merged_df["TripType"].unique())} unique trip sequences.")

    logging.debug(f"Finding the probability of every trip sequence for a given number of trips per day")
    trip_probs = merged_df[["SumTrips", "TripType"]].groupby("SumTrips").value_counts(normalize=True).reset_index()
    logging.debug(trip_probs.head())

    # Converting string of combinations back to list
    trip_probs["TripType"] = trip_probs["TripType"].apply(ast.literal_eval)

    logging.debug("Making a dictionary where key is number of trips and values are two lists")
    logging.debug("first entry is the trip combinations, second entry is their respective probabilities")

    pop_seq_weights = {}

    for trip_length in trip_probs.SumTrips.unique():
        by_trip_length = trip_probs[trip_probs["SumTrips"] == trip_length]
        logging.debug(f"Showing head when trip length = {trip_length}")
        logging.debug(by_trip_length.head())
        population = list(by_trip_length["TripType"])
        weights = list(by_trip_length["proportion"])
        #logging.debug(population)
        #logging.debug(weights)
        pop_seq_weights[f"trip_length_{int(trip_length)}"] = [population, weights]

    logging.debug("Showing highest sequence probabilities for trip lengths <=4")
    for i,(k,v) in enumerate(pop_seq_weights.items()):
        if i <=4:
            logging.debug(f"{k}, {v[0][0]}, {v[1][0]}")

    return pop_seq_weights

def return_copula(df, weekday, year, trip_cut_off, rare_threshold = 100, plots=False):

    """
     Generates a copula multivariate distribution for trip_start, trip_end and distance based on a trip number and trip type

    args:
        df(pd.DataFrame): Original travel dataframe from which to fit copula
        weekday(int): 1 for weekday and 2 for weekend values only#
        year(int): the year for which to filter the data
        trip_cut_off(int): Max trip length, any trips longer than this get assigned the max trip length

    Returns:
        dict: (trip_number, (trip type)) as key and fitted copula as value
    """    

    df = df.copy()

    df = filter_df(df, weekday, year)

    # Ensuring df is correctly filtered

    logging.debug(f"unique weekday: {df.head()["TravelWeekDay_B03ID"]}")
    logging.debug(f"Unique year: {df.head()["TravelYear"]}")

    if plots is True:
        # Work trips
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.title("A Histogram of Work Trip Start Times")
        plt.grid()
        work_home = df[df["TripType"] == (1,3)]
        home_work = df[df["TripType"] == (3,1)]
        plt.hist(work_home["TripStart"], bins=50, label="work->home")
        plt.hist(home_work["TripStart"], bins=50, label="home->work")
        plt.legend()
        
        
        plt.subplot(1,2,2)
        plt.title("A Histogram of Work Trip Distance Travelled")
        plt.grid()
        plt.hist(work_home["TripDisExSW"], bins=50, range=(0,60))

        plt.tight_layout()
        plt.show()

        # Home other trips
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.title("A Histogram of Home->Other Trip Start Times")
        plt.grid()
        home_other = df[df["TripType"] == (3,2)]
        plt.hist(home_other["TripStart"], bins=50)
            
        plt.subplot(1,2,2)
        plt.title("A Histogram of Home->Other trip Distance Travelled")
        plt.grid()
        plt.hist(home_other["TripDisExSW"], bins=50, range=(0,60))

        plt.tight_layout()
        plt.show()

    # Copulas
    logging.info("Using Copulas to generate multivariate distributions for continous variables")

    logging.debug("Creating column with 'JourSeq' 'TripType' pairs")
  
    # Making all journeys > 10 (cut-off) value == 10

    df["combos"] = [list(x) for x in list(zip(df["JourSeq"], df["TripType"]))]

    # All possible combos of "JourSeq" and "TripType"

    df["combos"] = df["combos"].apply(lambda x: [trip_cut_off  if x[0] > trip_cut_off else x[0]] + list(x[1:]))

    # Converting back to a tuple

    df["combos"] = df["combos"].apply(tuple)

    # Dropping Nans
    df = df.dropna(axis=0)

    ## Assign all rare combos to (999, (999,999)) dist

    # Obtaining calue counts
    combo_val_counts = df["combos"].value_counts()

    df["combos"] = df["combos"].map(lambda x: x if combo_val_counts[x] > rare_threshold else (999,(999,999)))

    logging.debug("All unique combos ...")
    logging.debug(f"{df["combos"].unique()}")

    # Using a general distribution for very rare journey sequences.

    #general_dist = GaussianMultivariate()
    #general_to_copula = df[["TripStart", "TripEnd", "TripDisExSW"]]
    #general_to_copula = general_to_copula.sample(n=10000)
    #general_dist.fit(general_to_copula)

    copulas = {}

    for combo in df["combos"].unique():
        logging.debug(f"Fitting combo: {combo}")
        dist = GaussianMultivariate()
        to_copula = df[df["combos"] == combo][["TripStart", "TripEnd", "TripDisExSW"]]
        logging.debug("Head of to_copulas...")
        logging.debug(f"{to_copula.head()}")
        # Dropping nans
        #to_copula = to_copula.dropna(axis=0)
        logging.debug("Having dropped nans")
        logging.debug(f"{to_copula.head()}")

        logging.debug("Taking 10000 randomly drawn samples or otherwise copula crashes")
        logging.debug(f"len of df before sampling: {len(to_copula)}")

        '''
        if len(to_copula) < 20:
            logging.debug("Small sample size, Falling back to general distribution")
            copulas[combo] = general_dist
        '''

        if len(to_copula) > 10000:
            to_copula = to_copula.sample(n=10000)

        logging.debug(f"len of df after sampling: {len(to_copula)}")   

        dist.fit(to_copula)
        copulas[combo] = dist
        logging.info(f"{combo} is complete!")

        logging.debug("Adding general distribution")

    #copulas["general_dist"] = general_dist
             
    return copulas

#c = return_trip_start_end(df=car_df, weekday=1, year=2014, plots=False)

#return_journey_seq(car_df, 1, 2014)

def gen_cont_seq(row, copula_dicts, restart_threshold=300):

    """ 

    Generates a (start_time, end_time, trip_distance) for every individual based on the day of the week, his trip sequence
    and the number of trips he did that day. Uses a copula to simulate continuous variables based on categorical distributions
    of number of trips per day, day of the week and trip sequence. Copula's are truncated as certain conditions must be satisfied. 
    I.E. all values must be positive, end time must be after start time and the start time of the next trip must be after the end time
    of the previous trip. The algorithm can get stuck if for example an individual has many trips and an early is randomly sampled at a 
    very late time. To circumvent this a restart threshold is set to restart the algorithm in the case that a solution is not found.
    With the restart threshold set to 300 it takes around a mimute to generate a year of travel for one individual.

    Parameters:
        row (pd.Series): A row of the a dataframe containing [["Year","we_wd","trip_num","trip_seqs"]], these should have been generated 
                         previously in .simulate()

        copula_dicts(list): A list of copula dictionaries, the first item MUST be the weekday copula nd the second item MUST be the weekend copula

        restart_threshold(int): The number of iterations before the search for a suitable trip_start, trip_end, distance combo is found for each
                                journey in an individual's sequence.

    Returns:
        list: a list of lists containing start_time, end_time, trip_distance
        
    """    
    logging.info(f"\n{row}")
    
    if isinstance(row["trip_seqs"], list):

        if len(copula_dicts) != 2 or type(copula_dicts) is not list:
            raise ValueError("copula_dicts must a be a list of length 2")

        if row["we_wd"] == 1:
            copula_dict = copula_dicts[0]

        if row["we_wd"] == 2:
            copula_dict = copula_dicts[1]

        logging.debug("A list of copula dict keys")
        logging.debug(f"{list(copula_dict.keys())}")

        restart_attempts = 0

        while True:

            start_end_dis = []

            copula_matches =  [(i+1, row["trip_seqs"][i]) for i in range(row["num_trips"])]

            logging.debug(f"Copula matches for this row: {copula_matches}")

            restart_flag = False

            for i,match in enumerate(copula_matches):

                if copula_dict.get(match, None) is None:
                    logging.debug(f"Match: {match} not found in copula dict. Falling back to rare distribution")

                copula_obj = copula_dict.get(match, copula_dict.get( (999,(999,999))  ))

                copula_samp = np.round(copula_obj.sample(1).to_numpy(), 2).flatten()

                search_iterations = 0


                if i == 0:
                    logging.info(f"i: {i}: seq: {match}")
                    while True:

                        copula_samp = np.round(copula_obj.sample(1).to_numpy(), 2).flatten()
                        #print(f"copula_samp: {copula_samp}")
                        # All entries are negative and trip end is greater than trip start

                        #search_iterations += 1
                        #print(f"\rSearch Iterations: {search_iterations}", end="", flush=True)

                        if np.all(copula_samp>0) and (copula_samp[1] > copula_samp[0]):
                            print("")
                            break

                    start_end_dis.append(copula_samp)

                    logging.info(f"current output: {start_end_dis}\n")

                else:
                    while True:

                        copula_samp = np.round(copula_obj.sample(1).to_numpy(), 2).flatten()

                        #print(f"copula_samp: {copula_samp}")
                        # All entries are negative and trip end is greater than trip start and next trip trip start is greater than last trips trip end

                        search_iterations += 1
                        print(f"\rSearch Iterations: {search_iterations} | Restart Attempts: {restart_attempts}", end="", flush=True)

                        if search_iterations == restart_threshold:
                            logging.info("\nThreshold exceeded! Restarting entire sequence...")
                            restart_attempts += 1
                            restart_flag = True
                            break  # Break inner loop to restart

                        if np.all(copula_samp>0) and (copula_samp[1] > copula_samp[0]) and (copula_samp[0] > start_end_dis[-1][1]):
                            print("")
                            break

                    if restart_flag:
                        break

                    start_end_dis.append(copula_samp)

                    logging.info(f"current output: {start_end_dis}\n")

            return start_end_dis
        
    else:
        return 0


