import pandas as pd
import config as cfg
import random

# Loading in df

full_df_path = cfg.root_folder + "/dataframes/day_trip_merge.pkl"



def generate_charger(x, home_charger_likelihood=0.96, work_charger_likelihood=0.62, public_charger_likelihood=0.17):
    if x == 3:
        return random.choices([1,0], weights=[home_charger_likelihood,1-home_charger_likelihood])[0]
    if x == 2:
        return random.choices([1,0], weights=[public_charger_likelihood,1-public_charger_likelihood])[0]
    else:
        return random.choices([1,0], weights=[work_charger_likelihood,1-work_charger_likelihood])[0]



def charging_logic(df, is_loaded=False, test_index=None):

    if not is_loaded:

        individual_ids = df["IndividualID"].unique()
    
        for i in individual_ids[:test_index]:

            i_df = df[df["IndividualID"]==i]

            if 1 in i_df["TravDay"].values:
                # If individual uses his car on the first travel day
                pass

            else:
                pass

            return i_df