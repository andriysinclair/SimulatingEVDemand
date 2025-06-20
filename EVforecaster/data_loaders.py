import config as cfg
import logging
import pandas as pd
import pickle

# Obtaining relevant paths
trip_data = cfg.root_folder + "/data/trip_eul_2002-2023.tab"
day_data = cfg.root_folder + "/data/day_eul_2002-2023.tab"
household_data =cfg.root_folder + "/data/household_eul_2002-2023.tab"

# Set up basic configuration for logging
logging.basicConfig(level=logging.DEBUG)


def trip_data_loader(survey_years,
                     output_file_name, 
                     chunksize = 100000,
                     trip_path = trip_data,
                     trip_cols_to_keep = cfg.trip_cols_to_keep, 
                     trip_purpouse_mapping = cfg.trip_purpouse_mapping,
                     trip_type_mapping = cfg.trip_type_mapping,
                     is_loaded = False) -> pd.DataFrame:

    # Loading in chunks to reduce cost

    if not is_loaded:

        survey_years = [str(s_y) for s_y in survey_years]

        merged_chunks = []

        for i,chunk in enumerate(pd.read_csv(trip_path, sep="\t", chunksize=chunksize, dtype=str)):

            chunk = chunk[chunk["MainMode_B04ID"] == "3"]
            chunk = chunk[chunk["SurveyYear"].isin(survey_years)]

            if not chunk.empty:

                chunk = chunk[trip_cols_to_keep]

                # Converting all columns to int type

                chunk = chunk.astype(float)

                merged_chunks.append(chunk)

                logging.debug(f"chunk {i} has been saved")


            else:
                        
                # year in survey_years not found in chunk
                logging.debug(f"chunk {i} has been skipped")

                continue

        
        # Merging all gathered chunks

        merged_df = pd.concat(merged_chunks, ignore_index=True)

        logging.debug(f"Total rows before dropping duplicates: {len(merged_df)}")
        logging.debug(f"Exact duplicate rows: {merged_df.duplicated().sum()}")

        # Dropping missing purpouse values and mapping purpouses

        merged_df = merged_df[  ~((merged_df["TripPurpFrom_B01ID"] == -8) & (merged_df["TripPurpFrom_B01ID"] == -10))  ]
        merged_df = merged_df[  ~((merged_df["TripPurpTo_B01ID"] == -8) & (merged_df["TripPurpTo_B01ID"] == -10))  ]

        merged_df["TripPurpFrom_B01ID"] = merged_df["TripPurpFrom_B01ID"].map(trip_purpouse_mapping)
        merged_df["TripPurpTo_B01ID"] = merged_df["TripPurpTo_B01ID"].map(trip_purpouse_mapping)

        # Making trip type column; From -> To

        merged_df["TripType"] = (
            merged_df["TripPurpFrom_B01ID"].astype(str) + "-" + merged_df["TripPurpTo_B01ID"].astype(str)
        )

        # Mapping...
        merged_df["TripType"] = merged_df["TripType"].map(trip_type_mapping)

        # Renaming purpouse columns

        merged_df = merged_df.rename(columns={"TripPurpFrom_B01ID": "TripStartLoc",
                                              "TripPurpTo_B01ID": "TripEndLoc"})
        
        # Check missing values

        # Sort values to a rational order

        merged_df = merged_df.sort_values(by=["IndividualID", "TravDay", "JourSeq", "TripStart", "TripEnd"], ascending=True)

        #  Creating rolling time series for each individual for trip start

        merged_df["TripStartRolling"] = (merged_df["TravDay"] - 1) * 24*60 + merged_df["TripStart"]

        # """" for trip end

        merged_df["TripEndRolling"] = (merged_df["TravDay"] - 1) * 24*60 + merged_df["TripEnd"]

        # Dumping to pickle

        with open(cfg.root_folder + f"/dataframes/{output_file_name}.pkl", "wb") as f:
            pickle.dump(merged_df, f)   

        # Dumping to csv
        merged_df.to_csv(cfg.root_folder + f"/output_csvs/{output_file_name}.csv", index=False)

        logging.debug(f"File saved to {output_file_name}")

        for col in merged_df.columns:
            logging.debug(col)

        print("")

        logging.debug(merged_df.dtypes)

        return merged_df
    
    if is_loaded:

        # If data has already been loaded. Loading pickle

        with open(cfg.root_folder + f"/dataframes/{output_file_name}.pkl", "rb") as f:
            merged_df = pickle.load(f)   

        logging.info(f"loading {output_file_name}")
    
        for col in merged_df.columns:
            logging.debug(col)

        print("")

        logging.debug(merged_df.dtypes)

        return merged_df
    
def day_data_loader( output_file_name, 
                     day_path = day_data,
                     day_cols_to_keep = cfg.day_cols_to_keep, 
                     is_loaded = False) -> pd.DataFrame:
    
    # Keeping only necessary cols

    if not is_loaded:

        day_df = pd.read_csv(day_path, sep="\t")

        day_df = day_df[day_cols_to_keep]

        # Converting day_ID to float for better merge
        day_df["DayID"] = day_df["DayID"].astype(float)

        logging.debug(day_df.dtypes)

        with open(cfg.root_folder + f"/dataframes/{output_file_name}.pkl", "wb") as f:
            pickle.dump(day_df, f)  

        day_df.to_csv(cfg.root_folder + f"/output_csvs/{output_file_name}.csv", index=False) 

        logging.debug(day_df.dtypes)

        logging.debug(f"File saved to {output_file_name}")

        return day_df
    
    if is_loaded:

        with open(cfg.root_folder + f"/dataframes/{output_file_name}.pkl", "rb") as f:
            day_df = pickle.load(f)   

        logging.debug(day_df.dtypes)

        logging.info(f"loading {output_file_name}")

        return day_df
    
def household_data_loader(output_file_name,
                          household_path = household_data,
                          household_cols_to_keep = cfg.household_cols_to_keep,
                          is_loaded=False):
    
    if not is_loaded:
        household_df = pd.read_csv(household_path, sep="\t")
        household_df = household_df[household_cols_to_keep]

        household_df["HouseholdID"] = household_df["HouseholdID"].astype(float)

        logging.debug(household_df.dtypes)

        with open(cfg.root_folder + f"/dataframes/{output_file_name}.pkl", "wb") as f:
            pickle.dump(household_df, f)   

        household_df.to_csv(cfg.root_folder + f"/output_csvs/{output_file_name}.csv", index=False)

        logging.debug(f"File saved to {output_file_name}")

        return household_df
    
    if is_loaded:

        with open(cfg.root_folder + f"/dataframes/{output_file_name}.pkl", "rb") as f:
            household_df = pickle.load(f)   

        logging.debug(household_df.dtypes)

        logging.info(f"loading {output_file_name}")

        return household_df

    
def merge_dfs(df1, df2, df3, travel_year, common_id_1_2, common_id_2_3, output_file_name, is_loaded=False) -> pd.DataFrame:
    """
    merge_dfs 

    Merges loaded and subsetted trip data, day data and household data

    Args:
        df1 (pd.DataFrame): Usually trip data
        df2 (pd.DataFrame): Usually day data
        df3 (pd.DataFrame): Usually household data
        travel_year (list): Travel years on which to subset
        common_id_1_2 (str): Usually "DayID"
        common_id_2_3 (str): Usually "HouseholdID"
        output_file_name (str): output file name (no suffix)
        is_loaded (bool, optional): Has dataset already been loaded?. Defaults to False.

    Returns:
        pd.DataFrame: Merged Dataframe
    """    
    if not is_loaded:
        df1 = df1.copy()
        df2 = df2.copy()
        df3 = df3.copy()

        merged_df_1_2 = pd.merge(left=df1, right=df2, on=common_id_1_2)

        merged_df_1_2 = merged_df_1_2[ merged_df_1_2["TravelYear"].isin(travel_year)  ]

        merged_df_1_2_3 = pd.merge(left=merged_df_1_2, right=df3, on=common_id_2_3)

        # Counting missing values

        missing_counts = merged_df_1_2_3.isna().sum()
        logging.debug("Missing value counts per column:")
        logging.debug("\n" + str(missing_counts))

        # Dropping all rows with missing values
        merged_df_1_2_3 = merged_df_1_2_3.dropna()

        # Making further transformations


        with open(cfg.root_folder + f"/dataframes/{output_file_name}.pkl", "wb") as f:
            pickle.dump(merged_df_1_2_3, f)   

        merged_df_1_2_3.to_csv(cfg.root_folder + f"/output_csvs/{output_file_name}.csv", index=False)

        logging.debug(f"File saved to {output_file_name}")



        return merged_df_1_2_3

    else:
        with open(cfg.root_folder + f"/dataframes/{output_file_name}.pkl", "rb") as f:
            merged_df_1_2_3 = pickle.load(f)  

        logging.info(f"loading {output_file_name}")

        return merged_df_1_2_3
    
def apply_preparatory(df, output_file_name):
    df = df.copy()

    individual_ids = df["IndividualID"].unique()

    # Here delete all weeks where TWSWeek = 53 as that is not a real week of the year

    df = df[df["TWSWeek"] != 53]

    df_by_i = []

    for i in individual_ids:

        i_df = df[df["IndividualID"]==i]
        i_df = i_df.copy()

        # Calculating time at trip end location

        # bring trip start rolling forward
        i_df["TripStartRolling+1"] = i_df["TripStartRolling"].shift(-1)

        # Calculate time at end location
        i_df["TimeEndLoc"] = i_df["TripStartRolling+1"] - i_df["TripEndRolling"] 

        i_df["Distance+1"] = i_df["TripDisExSW"].shift(-1)

        #Correct TWSweek to account for trips crossing over into new weeks. As TWSweek records the week the travel diary started
        # If travel starts on Sunday ....
        i_df["WeekDayDiff"] = i_df["TravelWeekDay_B01ID"].diff()

        i_df["WeekRollover"] = (i_df["WeekDayDiff"] < 0).astype(int)
        i_df["WeekRollover"] = i_df["WeekRollover"].cumsum()

        i_df["TWSWeekNew"] = i_df["TWSWeek"] + i_df["WeekRollover"]

        # Moving to January..

        i_df.loc[i_df["TWSWeekNew"] == 53, "TWSWeekNew"] = 1

        #logging.debug(i_df[["TravelWeekDay_B01ID", "WeekDayDiff", "WeekRollover", "TWSWeek", "TWSWeekNew"]])

        df_by_i.append(i_df)

    df = pd.concat(df_by_i)

    logging.info(f"Unique travel weeks new: {df["TWSWeekNew"].max()}")
    #logging.info(f"Unique travel weeks: {df["TWSWeek"].unique()}")
    logging.info(f"Unique travel year: {df["TravelYear"].unique()}")
    logging.info(df["TravelYear"].value_counts())

    df.to_pickle(cfg.root_folder + f"/dataframes/{output_file_name}.pkl")

    df.to_csv(cfg.root_folder + f"/output_csvs/{output_file_name}.csv", index=False)

    return df


def data_loader_end_to_end(travel_year, raw_data_frames_loaded=True):

    # Load in all your data frames
    if raw_data_frames_loaded is False:
        trip_df = trip_data_loader(survey_years=list(range(2012,2019)), output_file_name=f"trip_df_2012_2019", is_loaded=False)
        logging.info(f"Trip data loaded!")
        day_df = day_data_loader(output_file_name="day_df", is_loaded=False)
        logging.info("Day data loaded!")
        household_df = household_data_loader(output_file_name="household_df", is_loaded=False)
        logging.info(f"household data loaded")

        merged_df = merge_dfs(df1=trip_df, df2=day_df, df3=household_df, common_id_1_2="DayID", common_id_2_3="HouseholdID", travel_year=travel_year, output_file_name=f"merge_df_{travel_year}.pkl", is_loaded=False)
        logging.info(f"Datasets merged")

    if raw_data_frames_loaded:
        trip_df = pd.read_pickle(cfg.root_folder + f"/dataframes/trip_df_2016_2023.pkl")
        day_df = pd.read_pickle(cfg.root_folder + f"/dataframes/day_df.pkl")
        household_df = pd.read_pickle(cfg.root_folder + f"/dataframes/household_df.pkl")

        merged_df = merge_dfs(df1=trip_df, df2=day_df, df3=household_df, common_id_1_2="DayID", common_id_2_3="HouseholdID", travel_year=travel_year, output_file_name=f"merge_df_{travel_year}.pkl", is_loaded=False)
        logging.info(f"Datasets merged")

    

    final_df = apply_preparatory(merged_df, output_file_name=f"Ready_to_model_df_{travel_year}")

    logging.info(f"Final DF loaded")


    return final_df



if __name__ == "__main__":

    df = data_loader_end_to_end(travel_year=[2012,2013,2014,2015,2016,2017, 2018], raw_data_frames_loaded=True)


