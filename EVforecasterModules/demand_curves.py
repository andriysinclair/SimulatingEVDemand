import pandas as pd
import config as cfg
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import math
import logging

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO)

def output_full_long_df(df):
    """
    output_full_long_df 

    Adds necessary charging-related columns to charging_df that are needed to later make
    the wide transformation

    Args:
        df (pd.DataFrame): charging_df

    Returns:
        pd.DataFrame: complete df ready for wide transformation
    """    

    # Creating rolling columns from 0 - 10,080 (minutes in a week)

    df = df.copy()

    df["ChargeStartRolling"] = df["ChargeStart"] + (  (df["TravelDay"]-1)*1440  )
    df["ChargeEndRolling"] = df["ChargeEnd"] +    (  (df["TravelDay"]-1)*1440  )

    if "ChargeDuration" not in df.columns:
        df["ChargeDuration"] = df["ChargeEndRolling"] - df["ChargeStartRolling"]

    # Some charges roll over into the next week

    MAX_MINS = 7*1440 # 10,080 minutes in a week

    updated_rows = []

    for idx, trip in df.iterrows():
        if trip["ChargeEndRolling"] > MAX_MINS:
            # If a trip has gone into the next week

            overflow = trip["ChargeEndRolling"] - MAX_MINS
            


            trip_clipped = trip.copy()
            trip_clipped["ChargeEnd"] = 1440   # Max minutes in 24 hours
            
            trip_clipped["ChargeDuration"] = trip_clipped["ChargeEnd"] - trip_clipped["ChargeStart"]
            trip_clipped["ChargeEndRolling"] = MAX_MINS
            trip_clipped["TotalPowerUsed"] = trip_clipped["ChargeDuration"]/60 * trip_clipped["ChargingRate"]

            updated_rows.append(trip_clipped)

            # Create a new row
            trip_overflow = trip.copy()
            trip_overflow["ChargeStartRolling"] = 0
            trip_overflow["ChargeEndRolling"] = overflow
            trip_overflow["ChargeStart"] = 0
            trip_overflow["ChargeEnd"] = trip_overflow["ChargeEnd"] - 1440
            trip_overflow["ChargeDuration"] = trip_overflow["ChargeEnd"]

            # Increase travel week by 1
            trip_overflow["TravelWeek"] += 1
            trip_overflow["TravelDay"] = math.ceil(overflow/1440)
            trip_overflow["TotalPowerUsed"] = trip_overflow["ChargeDuration"]/60 * trip_overflow["ChargingRate"]

            updated_rows.append(trip_overflow)

        else:
            updated_rows.append(trip)

    df = pd.DataFrame(updated_rows).reset_index(drop=True)


    df["ChargeStartBin"] = df["ChargeStartRolling"]/5
    df["ChargeEndBin"] = df["ChargeEndRolling"]/5

    df["5_min_demand"] = df["ChargingRate"] / 60 * 5

    # Test wether we can get TotalPowerUsed again using the 5-minute power consumption

    df["MathTest"] = (df["ChargeEnd"] - df["ChargeStart"]) * df["5_min_demand"]/5

    df["MathMatch"] = np.isclose(df["MathTest"], df["TotalPowerUsed"], rtol=1e-1)

    # 3. Log mismatches if any
    if not df["MathMatch"].all():
        logging.debug(f"WARNING: Slight mathematical errors in calculation")
        logging.debug("❗Mathematical mismatch detected in some rows.")
        mismatches = df[~df["MathMatch"]][[
            "ChargeStart", "ChargeEnd", "TotalPowerUsed",
            "TravelDay", "ChargeStartRolling", "ChargeEndRolling",
            "ChargeStartBin", "ChargeEndBin", "5_min_demand", "MathTest"
        ]]
        logging.debug(f"{len(mismatches)} mismatches found")
        logging.debug("\n" + mismatches.to_string(index=False))

    #Assert all match


    #assert df["MathMatch"].all(), "Mismatch between calculated and actual TotalPowerUsed!"
    return df

def output_wide_df(df, location=[1,2,3]):

    df = df.copy()

    # Subsetting on location done here
    df = df[df["ChargeLoc"].isin(location)]

    # Build a blank df in wide format with the individual and each of his binned 5-minutly consumption

    bin_edges = np.arange(0, 7*1440+5, 5)

    #logging.debug(bin_edges)

    bin_labels = [f"{start}-{start+5}" for start in bin_edges[:-1]] 

    logging.debug(f"first 5 bin labels: {bin_labels[:5]}")
    logging.debug(f"final 5 bin lavels: {bin_labels[-5:]}")

    #long_df = pd.DataFrame(columns=["IndividualID"] + bin_labels)

    ####

    unique_is = df["IndividualID"].unique()

    all_rows = []

    for i in unique_is:

        i_df = df[df["IndividualID"] == i]

        # start a new row with all 0s
        
        demand_row = {label: 0 for label in bin_labels}
        demand_row["IndividualID"] = i

        for idx, trip in i_df.iterrows():

            #get intiger bin range for this trip
            logging.debug(f"Individual: {i}")
            start_bin = int(trip["ChargeStartBin"])
            logging.debug(f"Start bin: {start_bin}")
            end_bin = int(trip["ChargeEndBin"])
            logging.debug(f"End bin: {end_bin}")

            for b in range(start_bin, end_bin):
                bin_label = f"{b*5}-{b*5+5}"
                
                demand_row[bin_label] += trip["5_min_demand"]

            #logging.debug(f"bin label: {bin_label}")
            
        # append row to all rows
        all_rows.append(demand_row)


    wide_df = pd.DataFrame(all_rows)  

    return wide_df

def create_labels(df):
        
        df = df.copy()

        labels = df.columns[:-1]
        labels = [  int(label.split("-")[0]) for label in labels       ]
        labels_dow = [   math.ceil(label/1440)     for label in labels]
        labels_dow[0] = 1


        dow_mapping = {
                    1: "Mon",
                    2: "Tue",
                    3: "Wed",
                    4: "Thu",
                    5: "Fri",
                    6: "Sat",
                    7: "Sun"}
        
        labels_dow_mapped = [dow_mapping[label] for label in labels_dow]

        labels_hour = [label - (label_dow-1)*1440 for label, label_dow in zip(labels, labels_dow)]

        labels_hour = [f"{h:02d}:{m:02d}" for h,m in [divmod(mins,60) for mins in labels_hour]]

        new_labels = [f"{dow} {hm}" for dow, hm in zip(labels_dow_mapped, labels_hour)]
        
        logging.debug(labels[30:40])
        logging.debug(labels_dow[30:40])
        logging.debug(labels_dow_mapped[30:40])
        logging.debug(labels_hour[30:40])
        logging.debug(new_labels[30:40])

        return new_labels


if __name__ == "__main__":

    matplotlib.use("Agg")

    # Loading in df

    charging_df_path = cfg.root_folder + "/dataframes/charging_df.pkl"

    plots_folder = cfg.root_folder + "/plots/"
    charging_df = pd.read_pickle(charging_df_path)

    charging_df.to_csv(cfg.root_folder + "/output_csvs/charging_df.csv", index=False)

    #df, demand_df = output_demand_curves(charging_df=charging__df, suffix_long="demand_all_loc_all_week_long",
    #                                     suffix_wide="demand_all_loc_all_week_wide", is_loaded=True, plot=True)
