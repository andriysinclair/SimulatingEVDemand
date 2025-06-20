import config as cfg
import logging
import pandas as pd
import pickle


# Set up basic configuration for logging
logging.basicConfig(level=logging.DEBUG)


def trip_data_loader(survey_years,
                     trip_path,
                     trip_cols_to_keep, 
                     trip_purpouse_mapping,
                     trip_type_mapping,
                     chunksize = 100000) -> pd.DataFrame:

    # Loading in chunks to reduce cost

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

    # Dumping to pickle. Only if DEBUG mode

    return merged_df
    
def day_data_loader(  
                     day_path,
                     day_cols_to_keep) -> pd.DataFrame:
    
    # Keeping only necessary cols

    day_df = pd.read_csv(day_path, sep="\t")

    day_df = day_df[day_cols_to_keep]

    # Converting day_ID to float for better merge
    day_df["DayID"] = day_df["DayID"].astype(float)

    logging.debug(day_df.dtypes)

    logging.debug(day_df.dtypes)

    return day_df
    
    
def household_data_loader(
                          household_path,
                          household_cols_to_keep):
    
    household_df = pd.read_csv(household_path, sep="\t")
    household_df = household_df[household_cols_to_keep]

    household_df["HouseholdID"] = household_df["HouseholdID"].astype(float)

    logging.debug(household_df.dtypes)

    return household_df
    

def merge_dfs(df1, df2, df3, travel_year, common_id_1_2, common_id_2_3) -> pd.DataFrame:
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

    return df


def data_loader_end_to_end(travel_year, trip_path, day_path, household_path, survey_years,
                           trip_cols_to_keep, trip_purpouse_mapping, trip_type_mapping,
                           day_cols_to_keep, household_cols_to_keep):

    trip_df = trip_data_loader(survey_years=survey_years, 
                                trip_path=trip_path,
                                trip_cols_to_keep=trip_cols_to_keep,
                                trip_purpouse_mapping=trip_purpouse_mapping,
                                trip_type_mapping=trip_type_mapping,
                                )
    
    logging.info(f"Trip data loaded!")
    day_df = day_data_loader(day_path=day_path, day_cols_to_keep=day_cols_to_keep)
    logging.info("Day data loaded!")
    household_df = household_data_loader(household_path=household_path, household_cols_to_keep=household_cols_to_keep)
    logging.info(f"household data loaded")

    merged_df = merge_dfs(df1=trip_df, df2=day_df, df3=household_df, common_id_1_2="DayID", common_id_2_3="HouseholdID", travel_year=travel_year)
    logging.info(f"Datasets merged")

    final_df = apply_preparatory(merged_df, output_file_name=f"Ready_to_model_df_{travel_year}")

    logging.info(f"Final DF loaded")


    return final_df



if __name__ == "__main__":

    # Obtaining relevant paths
    trip_data = cfg.root_folder + "/data/trip_eul_2002-2023.tab"
    day_data = cfg.root_folder + "/data/day_eul_2002-2023.tab"
    household_data =cfg.root_folder + "/data/household_eul_2002-2023.tab"

    df = data_loader_end_to_end(travel_year=[2012,2013,2014,2015,2016,2017, 2018], raw_data_frames_loaded=True)


