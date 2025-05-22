import pandas as pd
import config as cfg
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import pickle
import math
import logging
from charging_logic_auxillary import generate_charger, obtain_decision_to_charge, calculate_charging_session

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO)

# Loading in df

charging_df_path = cfg.root_folder + "/dataframes/charging_df.pkl"
plots_folder = cfg.root_folder + "/plots/"
charging__df = pd.read_pickle(charging_df_path)

def plot_weekly_demand(demand_df, output_file_name):
    demand_df = demand_df.copy()

    # Removing individual id
    sums_over_interval = demand_df.iloc[:,:-1].sum()

    x = sums_over_interval.index
    y = sums_over_interval.values

    labels = demand_df.columns[:-1]
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

    plt.figure(figsize=(15,6))

    plt.title(f)

    plt.plot(x, y)

    plt.ylabel("Demand (kWh)")
    plt.xticks(ticks=range(0, len(new_labels), 72), labels=new_labels[::72], rotation=45)

    plt.tight_layout()

    plt.grid()


    plt.savefig(f"{plots_folder}{output_file_name}.pdf", format="pdf")



def output_demand_curves(charging_df, suffix_long, suffix_wide, location=[1,2,3], week_of_the_year = list(range(1,60)), is_loaded=False, plot=True  ):
    """
    output_demand_curves 

    Generate weekly charge demand curves (5-min intervals), averaged over individuals
    for a given week of the year. That is 7*1440 = 10,080 minutes OR 2016 5-min bins.


    Args:
        charging_df (pd.DataFrame): Charging schedule data frame generated from charging_logic.py
    """    
    if not is_loaded:
        # 1. Create a rolling ChargeStart ChargeEnd columns
        df = charging_df.copy()

        # Filter Data by location and by year
        df = df[df["ChargeLoc"].isin(location)]
        df = df[df["TravelWeek"].isin(week_of_the_year)]

        df["ChargeStartRolling"] = df["ChargeStart"] + (  (df["TravelDay"]-1)*1440  )
        df["ChargeEndRolling"] = df["ChargeEnd"] +    (  (df["TravelDay"]-1)*1440  )

        # Some charges roll over into the next week

        MAX_MINS = 7*1440 # 10,080 minutes in a week

        updated_rows = []

        for idx, trip in df.iterrows():
            if trip["ChargeEndRolling"] > MAX_MINS:
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
            logging.info(f"WARNING: Slight mathematical errors in calculation")
            logging.info("❗Mathematical mismatch detected in some rows.")
            mismatches = df[~df["MathMatch"]][[
                "ChargeStart", "ChargeEnd", "TotalPowerUsed",
                "TravelDay", "ChargeStartRolling", "ChargeEndRolling",
                "ChargeStartBin", "ChargeEndBin", "5_min_demand", "MathTest"
            ]]
            logging.info(f"{len(mismatches)} mismatches found")
            logging.info("\n" + mismatches.to_string(index=False))

        #Assert all match

        #assert df["MathMatch"].all(), "Mismatch between calculated and actual TotalPowerUsed!"

        df.to_csv(cfg.root_folder + f"/output_csvs/{suffix_long}.csv", index=False)
        df.to_pickle(cfg.root_folder + f"/dataframes/{suffix_long}.pkl")

        # Build a blank df in wide format with the individual and each of his binned 5-minutly consumption

        bin_edges = np.arange(0, 7*1440+5, 5)

        #logging.debug(bin_edges)

        bin_labels = [f"{start}-{start+5}" for start in bin_edges[:-1]] 

        logging.debug(f"first 5 bin labels: {bin_labels[:5]}")
        logging.debug(f"final 5 bin lavels: {bin_labels[-5:]}")

        demand_df = pd.DataFrame(columns=["IndividualID"] + bin_labels)

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


        demand_df = pd.DataFrame(all_rows)

        demand_df.to_csv(cfg.root_folder + f"/output_csvs/{suffix_wide}.csv", index=False)
        demand_df.to_pickle(cfg.root_folder + f"/dataframes/{suffix_wide}.pkl")

        return df,  demand_df
    
    if is_loaded:
        df = pd.read_pickle(cfg.root_folder + f"/dataframes/{suffix_long}.pkl")
        demand_df = pd.read_pickle(cfg.root_folder + f"/dataframes/{suffix_wide}.pkl")

        # Plot demand curves

    if plot:
        plot_weekly_demand(demand_df=demand_df, output_file_name=suffix_wide)

    
    return df, demand_df


if __name__ == "__main__":

    df, demand_df = output_demand_curves(charging_df=charging__df, suffix_long="demand_all_loc_all_week_long",
                                         suffix_wide="demand_all_loc_all_week_wide", is_loaded=True, plot=True)


