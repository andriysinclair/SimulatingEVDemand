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

    df = df.copy()

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

def output_wide_df(df, location=[1,2,3], week_of_the_year = list(range(1,60))):

    df = df.copy()

    # Subsetting done here
    df = df[df["ChargeLoc"].isin(location)]
    df = df[df["TravelWeek"].isin(week_of_the_year)]

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

def generate_plot(*args, travel_weeks_label, travel_year_label, total=False):

    location_mapping = {1: "Work",
                        2: "Other",
                        3: "Home"}

    for i, arg in enumerate(args):

        # Removing individual id
        sums_over_interval = arg.iloc[:,:-1].sum()

        x = sums_over_interval.index
        y = sums_over_interval.values

        labels = arg.columns[:-1]
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

        if not total:
            plt.plot(x, y, label=location_mapping[i+1])

        else:
            plt.plot(x, y, label="Total")

        
    if total:
        plt.ylabel("Demand (kWh)")


    plt.xticks(ticks=range(0, len(new_labels), 72), labels=new_labels[::72], rotation=45)
    plt.legend()

    if not total:
        plt.title(f"{travel_weeks_label} of {travel_year_label} by location.")

    if total:
        plt.title(f"{travel_weeks_label} of {travel_year_label} total.")

    plt.tight_layout()

    plt.grid()


def plot_weekly_demand(charging_df, output_file_name, week_of_the_year, week_label, year_label, save_fig = True):

    # Transform the charging df

    charging_df = charging_df.copy()


    long_df = output_full_long_df(charging_df)

    '''
    if len(long_df["TravelYear"].unique()) == 1:
        year_label = int(  long_df["TravelYear"].unique()[0]  )
    else:
        year_label = f"{int(long_df["TravelYear"].unique()[0])}-{int(long_df["TravelYear"].unique()[-1])}"

    if len(long_df["TravelWeek"].unique()) == 1:
        week_label = int(  long_df["TravelWeek"].unique()[0]  )
    else:
        week_label = f"{int(long_df["TravelWeek"].unique()[0])}-{int(long_df["TravelWeek"].unique()[-1])}"
    '''
    # Subset the charging df

    wide_df1 = output_wide_df(long_df, location=[1], week_of_the_year=week_of_the_year)
    wide_df2 = output_wide_df(long_df, location=[2], week_of_the_year=week_of_the_year)
    wide_df3 = output_wide_df(long_df, location=[3], week_of_the_year=week_of_the_year)
    wide_df_all = output_wide_df(long_df, week_of_the_year=week_of_the_year)

    plt.figure(figsize=(15,6))

    plt.subplot(1,2,1)
    generate_plot(wide_df_all, travel_weeks_label=week_label, travel_year_label=year_label, total=True)

    plt.subplot(1,2,2)
    generate_plot(wide_df1, wide_df2, wide_df3, travel_weeks_label=week_label, travel_year_label=year_label, total=False)

    plt.tight_layout()

    if save_fig:

        plt.savefig(f"{plots_folder}{output_file_name}.pdf", format="pdf")


if __name__ == "__main__":

    matplotlib.use("Agg")

    # Loading in df

    charging_df_path = cfg.root_folder + "/dataframes/charging_df.pkl"

    plots_folder = cfg.root_folder + "/plots/"
    charging_df = pd.read_pickle(charging_df_path)

    charging_df.to_csv(cfg.root_folder + "/output_csvs/charging_df.csv", index=False)

    #df, demand_df = output_demand_curves(charging_df=charging__df, suffix_long="demand_all_loc_all_week_long",
    #                                     suffix_wide="demand_all_loc_all_week_wide", is_loaded=True, plot=True)

    plot_weekly_demand(charging_df=charging_df, output_file_name="plot_total", week_of_the_year=list(range(2,53)), week_label="Full Year", year_label=2017)
    #plot_weekly_demand(charging_df=charging_df, output_file_name="plot_total", week_of_the_year=list(range(1,60)))

    weeks_winter = [49,50,51,52] + list(range(1,10))
    plot_weekly_demand(charging_df=charging_df, output_file_name="plot_winter", week_of_the_year=weeks_winter, week_label="Winter", year_label=2017)

    weeks_spring = list(range(10,22))
    plot_weekly_demand(charging_df=charging_df, output_file_name="plot_spring", week_of_the_year=weeks_spring, week_label="Spring", year_label=2017)

    weeks_summer = list(range(22,36))
    plot_weekly_demand(charging_df=charging_df, output_file_name="plot_summer", week_of_the_year=weeks_summer, week_label="Summer", year_label=2017)

    weeks_autumn = list(range(36,49))
    plot_weekly_demand(charging_df=charging_df, output_file_name="plot_autumn", week_of_the_year=weeks_autumn, week_label="Autumn", year_label=2017)

