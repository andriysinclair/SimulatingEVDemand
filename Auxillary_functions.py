import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configure basic logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

# Importing dataframe for experimentation

car_df = pd.read_pickle("/home/trapfishscott/Cambridge24.25/Energy_thesis/Data/df_car.pkl")

def return_num_journey_prob(df, weekday, year, plots=True):
    df = df.copy()
    plt.style.use("ggplot")

    if "TravelWeekDay_B03ID" not in df.columns:
        raise ValueError(f"'TravelWeekDay_B03ID' not in DataFrame columns. DataFrame columns: {df.columns}")

    if weekday == 1:
        df = df[df["TravelWeekDay_B03ID"]==1]

    elif weekday == 2:
        df = df[df["TravelWeekDay_B03ID"]==2]

    else:
        raise ValueError(f"Weekday is {weekday}. Must be int(1) for weekday or int(2) for weekend ")
    
    if plots not in {True, False}:
        raise ValueError(f"plots is {plots}. Must == True or False")
    
    if not (2002 <= year <= 2023):  # Correct range check
        raise ValueError(f"Year is {year}. Must be between 2002 and 2023.")

    logging.debug(f"Filtering where year={year}")
    df = df[df["TravelYear"] == year]
    
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

return_num_journey_prob(df=car_df, weekday=1, year=2023, plots=False)


