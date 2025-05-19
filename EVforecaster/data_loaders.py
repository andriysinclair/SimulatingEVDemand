import config as cfg
import logging
import pandas as pd
import pickle

# Obtaining relevant paths
trip_data = cfg.root_folder + "/data/trip_eul_2002-2023.tab"
day_data = cfg.root_folder + "/data/day_eul_2002-2023.tab"

# Set up basic configuration for logging
logging.basicConfig(level=logging.DEBUG)


def trip_data_loader(trip_path, survey_years, chunksize, 
                     output_file_name, 
                     trip_cols_to_keep = cfg.trip_cols_to_keep, 
                     trip_purpouse_mapping = cfg.trip_purpouse_mapping,
                     trip_type_mapping = cfg.trip_type_mapping,
                     is_loaded = False) -> pd.DataFrame:

    # Loading in chunks to reduce cost

    if not is_loaded:

        survey_years = [str(s_y) for s_y in survey_years]

        merged_chunks = []

        for i,trip_df in enumerate(pd.read_csv(trip_path, sep="\t", chunksize=chunksize, dtype=str)):
            
            if "3" in trip_df["MainMode_B04ID"].unique():

                # Filter by car trips only

                trip_df = trip_df[trip_df["MainMode_B04ID"] == "3"]

                logging.debug(f"years in chunk {i+1}: {trip_df["SurveyYear"].unique()}")
                
                for year in trip_df["SurveyYear"].unique():

                    # Filtering years to manage dataframe size
                    
                    if year in survey_years:

                        chunk = trip_df[trip_df["SurveyYear"].isin(survey_years)]

                        chunk = chunk[trip_cols_to_keep]

                        # Converting all columns to int type

                        chunk = chunk.astype(float)

                        merged_chunks.append(chunk)


                    else:
                        
                        # year in survey_years not found in chunk

                        continue

            else:
                
                # Car trips not found in chunk

                continue
        
        # Merging all gathered chunks

        merged_df = pd.concat(merged_chunks, ignore_index=True)

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

        # Sort values to a rational order

        merged_df = merged_df.sort_values(by=["IndividualID", "TravDay", "JourSeq", "TripStart", "TripEnd"], ascending=True)

        # Dumping to pickle

        with open(cfg.root_folder + f"/dataframes/{output_file_name}", "wb") as f:
            pickle.dump(merged_df, f)   

        logging.info(f"File saved to {output_file_name}")

        for col in merged_df.columns:
            logging.debug(col)

        print("")

        logging.debug(merged_df.dtypes)

        return merged_df
    
    if is_loaded:

        # If data has already been loaded. Loading pickle

        with open(cfg.root_folder + f"/dataframes/{output_file_name}", "rb") as f:
            merged_df = pickle.load(f)   

        logging.info(f"loading {output_file_name}")
    
        for col in merged_df.columns:
            logging.debug(col)

        print("")

        logging.debug(merged_df.dtypes)

        return merged_df
    
def day_data_loader(day_path, 
                     output_file_name, 
                     day_cols_to_keep = cfg.day_cols_to_keep, 
                     is_loaded = False) -> pd.DataFrame:
    
    # Keeping only necessary cols

    if not is_loaded:

        day_df = pd.read_csv(day_path, sep="\t")

        day_df = day_df[day_cols_to_keep]

        # Converting day_ID to float for better merge
        day_df["DayID"] = day_df["DayID"].astype(float)

        logging.debug(day_df.dtypes)

        with open(cfg.root_folder + f"/dataframes/{output_file_name}", "wb") as f:
            pickle.dump(day_df, f)   

        logging.debug(day_df.dtypes)

        logging.info(f"File saved to {output_file_name}")

        return day_df
    
    if is_loaded:

        with open(cfg.root_folder + f"/dataframes/{output_file_name}", "rb") as f:
            day_df = pickle.load(f)   

        logging.debug(day_df.dtypes)

        logging.info(f"loading {output_file_name}")

        return day_df
    
def merge_dfs(df1, df2, common_id, output_file_name, is_loaded=False):

    if not is_loaded:
        df1 = df1.copy()
        df2 = df2.copy()

        merged_df = pd.merge(left=df1, right=df2, on=common_id)

        with open(cfg.root_folder + f"/dataframes/{output_file_name}", "wb") as f:
            pickle.dump(merged_df, f)   

        logging.info(f"File saved to {output_file_name}")

    else:
        with open(cfg.root_folder + f"/dataframes/{output_file_name}", "rb") as f:
            merged_df = pickle.load(f)  

        logging.info(f"loading {output_file_name}")

        return merged_df

def create_charging_nodes(trip_df):
    
    df = trip_df.copy()

    # For each individual fill in missing travel weeks



if __name__ == "__main__":
    print(cfg.root_folder)
    print("")
    trip_df = trip_data_loader(trip_path=trip_data, survey_years=[2017], 
                               chunksize=100000, output_file_name="trip_df_2017.pkl", is_loaded=True)
    
    day_df = day_data_loader(day_path=day_data,
                             output_file_name="day_df.pkl",
                             is_loaded=True)

    
    trip_df.to_csv(cfg.root_folder + "/output_csvs/trip_df_2017.csv", index=False)

    trip_df.head()

    print(len(day_df))
    print(day_df.head())

    day_trip_merge = merge_dfs(df1=trip_df, df2=day_df, common_id="DayID", output_file_name="day_trip_merge.pkl", is_loaded=True)

    day_trip_merge.to_csv(cfg.root_folder + "/output_csvs/day_trip_merge_2017.csv", index=False)
