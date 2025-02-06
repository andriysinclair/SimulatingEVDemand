# Script to load the dataframe and send to pickle
import pandas as pd
from pathlib import Path
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Data folder

nts_data_folder = Path(__file__).resolve().parent / "Data/NTS_data/UKDA-5340-tab/tab"
logging.info(f"NTS data folder path: {nts_data_folder}")

def find_file(nts_data_folder = nts_data_folder, index=None):

    if index is None: # Perform search
        num_files = 0

        for i,x in enumerate(nts_data_folder.iterdir()):
            if "Zone.Identifier" not in x.name:
                logging.info(f"index: {i} | file: {x}")
                num_files +=1

        logging.info(f"No. of files: {num_files}")

    else:
        logging.debug(f"Index is not None")
        for i,x in enumerate(nts_data_folder.iterdir()):
            logging.debug(f"{i}, {type(i), index}")

            if i == index:

                return x.name
            
            
        logging.error(f"Index: {index} was not found in directory")

# Check if folder exists
if not nts_data_folder.exists():
    logging.error(f"Directory not found: {nts_data_folder}")

else: 
    # Search through files

    find_file()

    find_file(index=200)



# 121 trips_eul for trips data
# 84 day_eul for day data

# Extracting trips data

trips_data_path = nts_data_folder / find_file(index=121)
logging.info(f"Trips data path: {trips_data_path}")

# Extracting day table

day_data_path = nts_data_folder / find_file(index=84)
logging.info(f"Day data path: {day_data_path}")

logging.info("Preparing to convert to DataFrames\n")

df = pd.read_csv(trips_data_path, delimiter="\t", usecols=[
                                                            "MainMode_B04ID",
                                                            "TripDisExSW",
                                                            "JourSeq",
                                                            "TripPurpFrom_B01ID",
                                                            "TripPurpTo_B01ID",
                                                            "TripStart",
                                                            "TripEnd",
                                                            "DayID",
                                                            "IndividualID",
                                                            "TravDay"], )

# Extracting Day file

df1 = pd.read_csv(day_data_path, delimiter="\t", usecols=["DayID", "TravelWeekDay_B03ID", "TravelYear"], )

logging.debug(f"Trip data info: {df.info()}")

logging.debug(f"Day data info: {df1.info()}")

## Mapping for trip purpouse

trip_purpouse_mapping = {1:1,   # work
                         2:2,   # Other
                         3:2,
                         4:2,
                         5:2,
                         6:2,
                         7:2,
                         8:2,
                         9:2,
                         10:2,
                         11:2,
                         12:2,
                         13:2,
                         14:2,
                         15:2,
                         16:2,
                         17:2,
                         18:2,
                         19:2,
                         20:2,
                         21:2,
                         22:2,
                         23:3,   # Home
                         -8: 0, 
                        -10: 0,} # Strange

logging.info("Preparing to merge dataframes on DayID...")
df_joined = df.merge(df1, on="DayID")

logging.info("Filtering on only those trips that used a car...")
df_car = df_joined[df_joined["MainMode_B04ID"] == 3]

logging.info("Dropping 'MainMode_B04ID', which determined transport used...")
df_car = df_car.drop("MainMode_B04ID", axis=1)

logging.info("Mapping trip purpouse to home, work, other")
df_car["TripPurpFrom_B01ID"] = df_car["TripPurpFrom_B01ID"].map(trip_purpouse_mapping)
df_car["TripPurpTo_B01ID"] = df_car["TripPurpTo_B01ID"].map(trip_purpouse_mapping)

logging.info("Dropping undefined trip purpouses")
df_car = df_car[df_car["TripPurpFrom_B01ID"] != 0]
df_car = df_car[df_car["TripPurpTo_B01ID"] != 0]

logging.info("Attempting to pickle DataFrame and save to data folder...")
Data_folder = Path(__file__).resolve().parent / "Data"

if Data_folder.exists():
    logging.info(f"Saving DataFrame as pickle to {Data_folder}...")
    df_car.to_pickle(Data_folder / "df_car.pkl")