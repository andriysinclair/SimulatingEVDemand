import pandas as pd
import config as cfg
import random
import numpy as np
import pickle
import logging
from charging_logic_auxillary import generate_charger, obtain_decision_to_charge, calculate_charging_session

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO)

# Loading in df

charging_df_path = cfg.root_folder + "/dataframes/charging_df.pkl"
charging__df = pd.read_pickle(charging_df_path)

def output_demand_curves(charging_df, suffix_long, suffix_wide, location=[1,2,3], week_of_the_year = list(range(1,60)), is_loaded=False  ):
    """
    output_demand_curves 

    Generate weekly charge demand curves (5-min intervals), averaged over individuals
    for a given week of the year. That is 9*1440 = 10,080 minutes OR 2016 5-min bins.
    We multiply by 9 as sunday charges can roll over into next monday, tuesday

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

        bin_edges = np.arange(0, 9*1440+5, 5)

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

        return df, demand_df


if __name__ == "__main__":

    df, demand_df = output_demand_curves(charging_df=charging__df, suffix_long="demand_all_loc_all_week_long",
                                         suffix_wide="demand_all_loc_all_week_wide", is_loaded=False)

    
    
