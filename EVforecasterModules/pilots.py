import config as cfg
import pandas as pd
import math
import matplotlib.pyplot as plt
import logging
import numpy as np
import pickle

from EVforecasterModules.demand_curves import output_wide_df, create_labels

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO)

def demand_curves_ECA(ECA_df_path, results_folder, ECA_weeks=list(range(39,54))) -> pd.DataFrame:

    """
    modify_df 

    This function will add the necessary columns to the pilot df that are needed for the functions under data_loader to work
    and create weekly EV demand curves for cross-comparison

    Args:
        df (pd.DataFrame): df relating to pilot study
    """    

    df = pd.read_csv(ECA_df_path)



    df["ChargeLoc"] = 3

    # Count how many charges took >25kW
    df25 = df[df["Energy"]>25]

    count_25 = len(df25)
    percentage25 = count_25 / len(df) * 100

    logging.info(f"Percentage of charges taking >25kW of energy: {percentage25:.2f}%")

    # Merge the date and time columns into one datetime column
    df['StartDateTime'] = pd.to_datetime(
        df['StartDate'] + ' ' + df['StartTime'], format='%d/%m/%Y %H:%M:%S')
    
    df['EndDateTime'] = pd.to_datetime(
        df['EndDate'] + ' ' + df['EndTime'], format='%d/%m/%Y %H:%M:%S')
    
    df['StartDateTime'] = pd.to_datetime(
        df['StartDateTime'], 
        dayfirst=True,
        format='%d/%m/%Y %H:%M'
    )
    df['EndDateTime'] = pd.to_datetime(
        df['EndDateTime'], 
        dayfirst=True,
        format='%d/%m/%Y %H:%M'
    )

    df = df.drop(columns=["StartDate", "StartTime", "EndDate", "EndTime"], axis=1)

    # Obtaining the difference
    df["diff"] = df["EndDateTime"] - df["StartDateTime"]
    df["diff"] = df["diff"].dt.total_seconds() / 60
    df["diff"] = 5 * round(df["diff"]/5)
    df["diff"] = df["diff"].astype(int)

    # Applying charge cap so that we get charge time not plugged in time

    min_charging_rate = 3 # kWh

    df["MaxChargingDuration"] = df["Energy"]/min_charging_rate * 60 # in minutes

    # finding rows that violate this condition

    mask = df["diff"] > df["MaxChargingDuration"]

    # Show some of the erronous rows
    overshoot_df = df.loc[mask, ["Energy", "StartDateTime", "EndDateTime", "diff", "MaxChargingDuration"]]

    logging.info(f"Rows where plug in duration was longer than charging duration")

    logging.info(overshoot_df.head())

    df.loc[mask, "diff"] = df.loc[mask, "MaxChargingDuration"].astype(int)

    # Getting charge start and charge end in minutes from midnight
    total_m_s = df["StartDateTime"].dt.hour * 60 + df["StartDateTime"].dt.minute
    total_m_s = 5 * round(total_m_s/5)
    df["ChargeStart"] = total_m_s
    df["ChargeStart"] = df["ChargeStart"].astype(int)
    df["ChargeEnd"] = df["ChargeStart"] + df["diff"]

    # Getting day of the week and week of the year
    df["TravelDay"] = df["StartDateTime"].dt.day_of_week+1
    df["TravelWeek"] = df["StartDateTime"].dt.isocalendar().week

    df = df[df["TravelWeek"].isin(ECA_weeks)]

    logging.info(f"Included travel weeks: {df["TravelWeek"].unique()}")

    df = df.sample(n=min(100_000, len(df)))

    # Dropping bits
    df = df.drop(["diff", "PluginDuration", "MaxChargingDuration"], axis=1)

    df["ChargeStartRolling"] = df["ChargeStart"] + (  (df["TravelDay"]-1)*1440  )
    df["ChargeEndRolling"] = df["ChargeEnd"] +    (  (df["TravelDay"]-1)*1440  )
    df["ChargeDuration"] = df["ChargeEndRolling"] - df["ChargeStartRolling"]

    df = df.rename(columns={"CPID": "IndividualID", 
                            "Energy": "TotalPowerUsed"})
    
    MAX_MINS = 7*1440 # 10,080 minutes in a week

    updated_rows = []

    for idx, trip in df.iterrows():
        if trip["ChargeEndRolling"] > MAX_MINS:
            # If a trip has gone into the next week

            overflow = trip["ChargeEndRolling"] - MAX_MINS

            charging_rate = trip["TotalPowerUsed"] / trip["ChargeDuration"]
            
            trip_clipped = trip.copy()
            trip_clipped["ChargeEnd"] = 1440   # Max minutes in 24 hours
            
            trip_clipped["ChargeDuration"] = trip_clipped["ChargeEnd"] - trip_clipped["ChargeStart"]
            trip_clipped["ChargeEndRolling"] = MAX_MINS
            trip_clipped["TotalPowerUsed"] = trip_clipped["ChargeDuration"] * charging_rate

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
            trip_overflow["TotalPowerUsed"] = trip_overflow["ChargeDuration"] * charging_rate

            updated_rows.append(trip_overflow)

        else:
            updated_rows.append(trip)

    df = pd.DataFrame(updated_rows).reset_index(drop=True)

    # Drop travel weeks that get moved to 53

    df = df[df["TravelWeek"] != 53]


    df["ChargeStartBin"] = df["ChargeStartRolling"]/5
    df["ChargeEndBin"] = df["ChargeEndRolling"]/5

    # To avoid 0 division error
    denominator = df["ChargeEndBin"] - df["ChargeStartBin"]
    df["5_min_demand"] = np.where(
        denominator > 0,
        df["TotalPowerUsed"] / denominator,
        np.nan
    )

    logging.info(f"Average 5 min demand: {df["5_min_demand"].mean()}")
    
    # Filtering by week of the year



    wide_df = output_wide_df(df)

    num_i  = len(wide_df)

    # Obtain the 5-min weekly demand vector
    demand_vector = wide_df.iloc[:,:-1].sum()

    y = demand_vector.values / num_i

    # Saving output vector

    return y



if __name__ == "__main__":

    import config as cfg

    csv_folder = cfg.root_folder + "/output_csvs/"
    plots_folder = cfg.root_folder + "/plots/"
    results_folder = cfg.root_folder + "/results/"

    # path to electric charge point analysis
    df_path = cfg.root_folder + "/data/electric-chargepoint-analysis-2017-raw-domestics-data.csv"

    df = pd.read_csv(df_path)

    print(df.columns)

