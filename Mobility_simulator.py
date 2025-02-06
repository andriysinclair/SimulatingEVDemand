import pandas as pd
import logging
from pathlib import Path
from Auxillary_functions import return_num_journey_prob, return_journey_seq
import random

# Configure basic logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)




# nts_df
df = pd.read_pickle("/home/trapfishscott/Cambridge24.25/Energy_thesis/Data/df_car.pkl")


class MobilitySimulator:

    def __init__(self, nts_df, year):

        self.mobility_schedule = pd.DataFrame()

        self.cols_to_use = set(['DayID', 'IndividualID', 'TravDay', 'JourSeq', 'TripPurpFrom_B01ID',
       'TripPurpTo_B01ID', 'TripStart', 'TripEnd', 'TripDisExSW', 'TravelYear',
       'TravelWeekDay_B03ID'])

        self.nts_df = nts_df

        if not isinstance(nts_df, pd.DataFrame):
            raise TypeError(f"Expected a DataFrame, got a {type(nts_df)} instead.")
        
        col_similarity = len(set(nts_df.columns) -  set(self.cols_to_use))

        if  col_similarity != 0:
            logging.error(f"There are {col_similarity} unknown columns")
            raise ValueError(f"DataFrame columns: {nts_df.columns}. Expected: {self.cols_to_use}. Unknown columns: {set(nts_df.columns) -  set(self.cols_to_use)}")

        logging.debug(f"Head of nts_df: {nts_df.head()}")

        self.year = year

        if year not in list(range(2002,2025)) and type(year)!=int:
            raise ValueError(f"year parameter: {year}, of type: {type(year)}. Must be in the range: 2002 <= year <= 2024 and of type int")
             

    def gen_ts(self):
        
        ts = pd.date_range(start=f"{self.year}-01-01", end=f"{self.year+1}-01-01")

        logging.debug(f"Length of time series: {len(ts)}")
        logging.debug(f"First value: {ts[0]}")
        logging.debug(f"Last value: {ts[-1]}\n")

        # Mapping weekday to 1 and weekend to 2 as in the datset
        we_wd_mapping = {0: 1,
                            1: 1,
                            2: 1,
                            3: 1,
                            4: 1,
                            5: 2,
                            6: 2}

        # Obtaining the year of the simulation

        year_part = ts.year
        we_wd_part = ts.weekday.map(we_wd_mapping)

        # Appending to mobility schedule:
        self.mobility_schedule["Year"] = year_part
        self.mobility_schedule["we_wd"] = we_wd_part

        logging.debug(f"Head of mobility_schedule: {self.mobility_schedule}")

        return self.mobility_schedule

    def simulate(self):
        ts =  self.gen_ts()
        logging.debug(f"Head of time series: {ts}")

        p_vector_wd = return_num_journey_prob(df=self.nts_df, weekday=1, year=self.year, plots=False)["p_vec"]
        p_vector_we = return_num_journey_prob(df=self.nts_df, weekday=2, year=self.year, plots=False)["p_vec"]
        sequence_prob_dict_1 = return_journey_seq(df=self.nts_df, weekday=1, year=self.year)
        sequence_prob_dict_2 = return_journey_seq(df=self.nts_df, weekday=2, year=self.year)

        trip_counts = list(range(0,11))

        ts["num_trips"] = ts["we_wd"].apply(lambda x: random.choices(trip_counts, p_vector_wd)[0] if x == 1 else random.choices(trip_counts, p_vector_we)[0])

        logging.debug(f"Head of time series with randomly generated trip numbs: {ts}")

        wd_trips = ts[ts["we_wd"] == 1]
        we_trips = ts[ts["we_wd"] == 2]

        perc_counts_wd = wd_trips["num_trips"].value_counts(normalize=True)
        perc_counts_we = we_trips["num_trips"].value_counts(normalize=True)

        logging.info(f"Percentage of num_trips (wd) in new series: {perc_counts_wd}")
        logging.info(f"Percentage of num_trips (we) in new series: {perc_counts_we}")

        self.mobility_schedule = ts

m = MobilitySimulator(nts_df=df, year=2021)

m.simulate()


print(m.mobility_schedule)
