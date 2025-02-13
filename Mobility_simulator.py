import pandas as pd
import logging
from pathlib import Path
from Auxillary_functions import return_num_journey_prob, return_journey_seq, return_copula, gen_cont_seq
import random

# Configure basic logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)




# nts_df
df = pd.read_pickle("/home/trapfishscott/Cambridge24.25/Energy_thesis/Data/df_car.pkl")


class MobilitySimulator:

    def __init__(self, nts_df, year, trip_cut_off):

        self.mobility_schedule = pd.DataFrame()

        self.cols_to_use = set(['DayID', 'IndividualID', 'TravDay', 'JourSeq', 'TripType', 'TripStart', 'TripEnd', 'TripDisExSW', 'TravelYear',
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
        
        self.trip_cut_off = trip_cut_off

        if trip_cut_off not in range(5,16):
            raise ValueError(f"trip_cut_off must be between 5 and 15. It is {trip_cut_off}")
        
        # Storing all tools needed to simulate data for final user (probabilities, copulas etc...)

        self.num_journey_p_vector_we = None
        self.num_journey_p_vector_wd = None

        self.jour_seq_p_vector_we = None
        self.jour_seq_p_vector_wd = None

        self.copulas_we = None
        self.copulas_wd = None
             

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
    
    def get_num_journey_p_vector(self):
        self.num_journey_p_vector_wd = return_num_journey_prob(df=self.nts_df, weekday=1, year=self.year, plots=False)["p_vec"]
        self.num_journey_p_vector_wd = return_num_journey_prob(df=self.nts_df, weekday=2, year=self.year, plots=False)["p_vec"]

    def gen_jour_seq_p_vector(self):
        self.jour_seq_p_vector_wd = return_journey_seq(df=self.nts_df, weekday=1, year=self.year)
        self.jour_seq_p_vector_we = return_journey_seq(df=self.nts_df, weekday=2, year=self.year)

    def gen_copulas(self):
        self.copulas_wd = return_copula(df=self.nts_df, weekday=1, year=self.year, trip_cut_off=10)
        self.copulas_we = return_copula(df=self.nts_df, weekday=2, year=self.year, trip_cut_off=10)

    def fit_tools(self):
        self.get_num_journey_p_vector()
        self.gen_jour_seq_p_vector()
        self.gen_copulas()

    def simulate(self):

        # Start Fresh

        self.mobility_schedule = pd.DataFrame()

        # Generate TS

        self.gen_ts()

        # Generate num trips

        trip_counts = list(range(0,self.trip_cut_off+1))

        self.mobility_schedule["num_trips"] = self.mobility_schedule["we_wd"].apply(lambda x: random.choices(trip_counts, self.num_journey_p_vector_wd)[0] if x == 1 else random.choices(trip_counts, self.num_journey_p_vector_we)[0])

        # Generate journey sequences

        self.mobility_schedule["trip_seqs"] = self.mobility_schedule.apply(
            lambda row: random.choices(
                self.jour_seq_p_vector_wd[f"trip_length_{row['num_trips']}"][0],  # List of sequences
                weights=self.jour_seq_p_vector_wd[f"trip_length_{row['num_trips']}"][1]  # Corresponding probabilities
            )[0] if row["we_wd"] == 1 else random.choices(
                self.jour_seq_p_vector_we[f"trip_length_{row['num_trips']}"][0],
                weights=self.jour_seq_p_vector_we[f"trip_length_{row['num_trips']}"][1]
            )[0],
            axis=1  # Apply row-wise
        )

        # Generate continous values based on copulas

        self.mobility_schedule["start_end_distance"] = self.mobility_schedule.apply(lambda row: gen_cont_seq(row, copula_dicts = [self.copulas_wd, self.copulas_we], restart_threshold=300), axis=1)



        return self.mobility_schedule


        #ts =  self.gen_ts()
        #logging.debug(f"Head of time series: {ts}")

        #p_vector_wd = return_num_journey_prob(df=self.nts_df, weekday=1, year=self.year, plots=False)["p_vec"]
        #p_vector_we = return_num_journey_prob(df=self.nts_df, weekday=2, year=self.year, plots=False)["p_vec"]
        #sequence_prob_dict_1 = return_journey_seq(df=self.nts_df, weekday=1, year=self.year)
        #sequence_prob_dict_2 = return_journey_seq(df=self.nts_df, weekday=2, year=self.year)

        #trip_counts = list(range(0,11))

        #ts["num_trips"] = ts["we_wd"].apply(lambda x: random.choices(trip_counts, p_vector_wd)[0] if x == 1 else random.choices(trip_counts, p_vector_we)[0])

        '''
        ts["trip_seqs"] = ts.apply(
            lambda row: random.choices(
                sequence_prob_dict_1[f"trip_length_{row['num_trips']}"][0],  # List of sequences
                weights=sequence_prob_dict_1[f"trip_length_{row['num_trips']}"][1]  # Corresponding probabilities
            )[0] if row["we_wd"] == 1 else random.choices(
                sequence_prob_dict_2[f"trip_length_{row['num_trips']}"][0],
                weights=sequence_prob_dict_2[f"trip_length_{row['num_trips']}"][1]
            )[0],
            axis=1  # Apply row-wise
        )
        '''

        #logging.debug(f"Head of time series with randomly generated trip numbs: {ts}")

        #wd_trips = ts[ts["we_wd"] == 1]
        #we_trips = ts[ts["we_wd"] == 2]

        #perc_counts_wd = wd_trips["num_trips"].value_counts(normalize=True)
        #perc_counts_we = we_trips["num_trips"].value_counts(normalize=True)

        #logging.info(f"Percentage of num_trips (wd) in new series: {perc_counts_wd}")
        #logging.info(f"Percentage of num_trips (we) in new series: {perc_counts_we}")





#m = MobilitySimulator(nts_df=df, year=2021, trip_cut_off=10)

#m.gen_ts()

#m.simulate_num_journey()

#m.simulate_jour_seq()

#m.simulate()


#print(m.mobility_schedule)
