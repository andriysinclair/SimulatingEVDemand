import config as cfg
import pandas as pd
import math
import matplotlib.pyplot as plt
import logging
import numpy as np

from demand_curves import output_wide_df, generate_plot

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO)

def demand_curves_MEA(df, week_of_the_year, week_label, year_label, save_fig, plots_folder, output_file_name):
    """
    modify_df 

    This function will add the necessary columns to the pilot df that are needed for the functions under data_loader to work
    and create weekly EV demand curves for cross-comparison

    Args:
        df (pd.DataFrame): df relating to pilor study
    """    
    df = df.copy()
    df["BatteryChargeStartDate"] = pd.to_datetime(df["BatteryChargeStartDate"])
    df["BatteryChargeStopDate"] = pd.to_datetime(df["BatteryChargeStopDate"])
    df["diff"] = df["BatteryChargeStopDate"] - df["BatteryChargeStartDate"]
    df["diff"] = df["diff"].dt.total_seconds() / 60
    df["diff"] = 5 * round(df["diff"]/5)
    df["diff"] = df["diff"].astype(int)

    total_m_s = df["BatteryChargeStartDate"].dt.hour * 60 + df["BatteryChargeStartDate"].dt.minute
    total_m_s = 5 * round(total_m_s/5)
    df["ChargeStart"] = total_m_s
    df["ChargeStart"] = df["ChargeStart"].astype(int)

    df["ChargeEnd"] = df["ChargeStart"] + df["diff"]
    df["TravelDay"] = df["BatteryChargeStartDate"].dt.day_of_week+1


    #df["ChargingRate"] = 3.6

    df["ChargeLoc"] = df["ParticipantID"].apply(lambda x: 1 if x[:2] == "YH" else 3)

    df["TravelWeek"] = df["BatteryChargeStartDate"].dt.isocalendar().week

    df = df.drop(["diff", "BatteryChargeStartDate", "BatteryChargeStopDate"], axis=1)

    df["TotalPowerUsed"] = 25 * df["Ending SoC (of 12)"]/12 -  25 * df["Starting SoC (of 12)"]/12

    df["ChargeStartRolling"] = df["ChargeStart"] + (  (df["TravelDay"]-1)*1440  )
    df["ChargeEndRolling"] = df["ChargeEnd"] +    (  (df["TravelDay"]-1)*1440  )
    df["ChargeDuration"] = df["ChargeEndRolling"] - df["ChargeStartRolling"]

    df = df.rename(columns={"ParticipantID": "IndividualID"})

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


    df["ChargeStartBin"] = df["ChargeStartRolling"]/5
    df["ChargeEndBin"] = df["ChargeEndRolling"]/5

    # To avoid 0 division error
    denominator = df["ChargeEndBin"] - df["ChargeStartBin"]
    df["5_min_demand"] = np.where(
        denominator > 0,
        df["TotalPowerUsed"] / denominator,
        np.nan
    )

    # Checking for any faulty charges
    df.loc[df["Starting SoC (of 12)"] == df["Ending SoC (of 12)"], "5_min_demand"] = np.nan


    logging.info(f"Average 5 min demand: {df["5_min_demand"].mean()}")

    #wide_df1 = output_wide_df(df, location=[1], week_of_the_year=week_of_the_year)
    #wide_df3 = output_wide_df(df, location=[3], week_of_the_year=week_of_the_year)
    wide_df_all = output_wide_df(df, week_of_the_year=week_of_the_year)

    plt.figure(figsize=(15,6))

    plt.subplot(1,2,1)
    generate_plot(wide_df_all, travel_weeks_label=week_label, travel_year_label=year_label, total=True)

    plt.subplot(1,2,2)

    locations = df["ChargeLoc"].unique()

    for loc in locations:
        wide_df_loc = output_wide_df(df, location=[loc], week_of_the_year=week_of_the_year)
        generate_plot(wide_df_loc, travel_weeks_label=week_label, travel_year_label=year_label, location=loc, total=False)


    plt.tight_layout()

    if save_fig:

        plt.savefig(f"{plots_folder}{output_file_name}.pdf", format="pdf")



    return df




    #df["ChargeStart"] = 



if __name__ == "__main__":

    import config as cfg
    
    pilot_data = cfg.root_folder + "/pilot_data/"
    plots_folder = cfg.root_folder + "/plots/"
    df = pd.read_csv(pilot_data + "MyElectricAvenue.csv")

    df1 = demand_curves_MEA(df,week_of_the_year=list(range(2,53)), week_label="Full Year", year_label=2017, plots_folder=plots_folder,
                            output_file_name="MEA_plot", save_fig=True)

    #df1 = output_full_long_df(df)

    df1.to_csv(cfg.root_folder + "/output_csvs/MEA_long.csv", index=False)