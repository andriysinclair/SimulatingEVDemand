### An end-to-end pipeline ###

# Requires trip-day-household merged dataframe to be loaded as pickle

#1. Create charging schedules - elements of randomisation introduced - charging availability, intial SOC, charge decisions etc...
#   Monte Carlo Approach


#2. Perform the necessary manipulations needed before creating the wide df - ChargeStartRolling, ChargeEndRolling etc.

#3. Subset the df from #2 on desired week of the year/ charge location

### Libraries ###

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
from charging_logic import charging_logic
from demand_curves import output_full_long_df, output_wide_df

### Path to merged df ###

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO)

# Loading in df

full_df_path = cfg.root_folder + "/dataframes/merge_trip_day_hh_2017.pkl"
full_df = pd.read_pickle(full_df_path)

charging_df_path = cfg.root_folder + "/dataframes/charging_df.pkl"
charging__df = pd.read_pickle(charging_df_path)

plots_folder = cfg.root_folder + "/plots/"

# Have all your subsetting done here
long_df = output_full_long_df(charging__df)

wide_df = output_wide_df(long_df)


