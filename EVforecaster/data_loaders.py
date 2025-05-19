import config as cfg
import logging
import pandas as pd
import pickle

# Obtaining relevant paths
trip_data = cfg.root_folder + "/data/trip_eul_2002-2023.tab"

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

        # Dropping legact columns

        merged_df = merged_df.drop(columns=["TripPurpFrom_B01ID", "TripPurpTo_B01ID"], axis=1)

        # Sort values to a rational order

        merged_df = merged_df.sort_values(by=["IndividualID", "TravDay", "JourSeq", "TripStart", "TripEnd"], ascending=True)

        # Dumping to pickle

        with open(cfg.root_folder + f"/dataframes/{output_file_name}", "wb") as f:
            pickle.dump(merged_df, f)   

        logging.info(f"File saved to {output_file_name}")

        return merged_df
    
    if is_loaded:

        # If data has already been loaded. Loading pickle

        with open(cfg.root_folder + f"/dataframes/{output_file_name}", "rb") as f:
            merged_df = pickle.load(f)   

        logging.info(f"loading {output_file_name}")

        return merged_df

def create_charging_nodes(trip_df):
    pass

if __name__ == "__main__":
    print(cfg.root_folder)
    print("")
    trip_df = trip_data_loader(trip_path=trip_data, survey_years=[2017], 
                               chunksize=100000, output_file_name="trip_df_2017.pkl", is_loaded=False)

    trip_df.to_csv(cfg.root_folder + "/output_csvs/trip_df_2017.csv", index=False)

    for col in trip_df.columns:
        logging.info(col)

    print("")

    print(trip_df.dtypes)