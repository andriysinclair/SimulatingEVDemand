import pandas as pd
import logging
from pathlib import Path
from Auxillary_functions import return_num_journey_prob, return_journey_seq, return_copula, gen_cont_seq
import random
import pickle

# Configure basic logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

log_file = "/home/trapfishscott/Cambridge24.25/Energy_thesis/logs/debug_log.log"
log_format = "%(asctime)s - %(levelname)s - %(message)s"

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(log_format))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(log_format))

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Optional: Remove duplicate logs if the root logger already has handlers
if logger.hasHandlers():
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# nts_df
df = pd.read_pickle("/home/trapfishscott/Cambridge24.25/Energy_thesis/Data/df_car.pkl")

# Tools folder
tools = str(Path(__file__).resolve().parent / "Tools")

logging.info(f"Tools folder: {tools}")

class MobilitySimulator:

    def __init__(self, year):

        self.mobility_schedule = pd.DataFrame()

        self.year = year

        if year not in list(range(2002,2025)) and type(year)!=int:
            raise ValueError(f"year parameter: {year}, of type: {type(year)}. Must be in the range: 2002 <= year <= 2024 and of type int")
                
        self.cols_to_use = set(['DayID', 'IndividualID', 'TravDay', 'JourSeq', 'TripType', 'TripStart', 'TripEnd', 'TripDisExSW', 'TravelYear',
       'TravelWeekDay_B03ID'])
        
        self.trip_cut_off = 10 # A value under 15 is recommended
        
        if self.trip_cut_off not in range(5,16):
            raise ValueError(f"trip_cut_off must be between 5 and 15. It is {self.trip_cut_off}")
                
        # Storing all tools needed to simulate data for final user (probabilities, copulas etc...)

        self.num_journey_p_vector_we = None
        self.num_journey_p_vector_wd = None

        self.jour_seq_p_vector_we = None
        self.jour_seq_p_vector_wd = None

        self.copulas_we = None
        self.copulas_wd = None

    def _gen_ts(self):
        
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
    
    def _get_num_journey_p_vector(self, nts_df):

        if not isinstance(nts_df, pd.DataFrame):
            raise TypeError(f"Expected a DataFrame, got a {type(nts_df)} instead.")
        
        col_similarity = len(set(nts_df.columns) -  set(self.cols_to_use))

        if  col_similarity != 0:
            logging.error(f"There are {col_similarity} unknown columns")
            raise ValueError(f"DataFrame columns: {nts_df.columns}. Expected: {self.cols_to_use}. Unknown columns: {set(nts_df.columns) -  set(self.cols_to_use)}")

        logging.debug(f"Head of nts_df: {nts_df.head()}")
        self.num_journey_p_vector_wd = return_num_journey_prob(df=nts_df, weekday=1, year=self.year, plots=False)["p_vec"]
        self.num_journey_p_vector_wd = return_num_journey_prob(df=nts_df, weekday=2, year=self.year, plots=False)["p_vec"]

    def _gen_jour_seq_p_vector(self, nts_df):

        if not isinstance(nts_df, pd.DataFrame):
            raise TypeError(f"Expected a DataFrame, got a {type(nts_df)} instead.")
        
        col_similarity = len(set(nts_df.columns) -  set(self.cols_to_use))

        if  col_similarity != 0:
            logging.error(f"There are {col_similarity} unknown columns")
            raise ValueError(f"DataFrame columns: {nts_df.columns}. Expected: {self.cols_to_use}. Unknown columns: {set(nts_df.columns) -  set(self.cols_to_use)}")

        logging.debug(f"Head of nts_df: {nts_df.head()}")

        self.jour_seq_p_vector_wd = return_journey_seq(df=nts_df, weekday=1, year=self.year)
        self.jour_seq_p_vector_we = return_journey_seq(df=nts_df, weekday=2, year=self.year)

    def _gen_copulas(self, nts_df):

        if not isinstance(nts_df, pd.DataFrame):
            raise TypeError(f"Expected a DataFrame, got a {type(nts_df)} instead.")
        
        col_similarity = len(set(nts_df.columns) -  set(self.cols_to_use))

        if  col_similarity != 0:
            logging.error(f"There are {col_similarity} unknown columns")
            raise ValueError(f"DataFrame columns: {nts_df.columns}. Expected: {self.cols_to_use}. Unknown columns: {set(nts_df.columns) -  set(self.cols_to_use)}")

        logging.debug(f"Head of nts_df: {nts_df.head()}")

        self.copulas_wd = return_copula(df=nts_df, weekday=1, year=self.year, trip_cut_off=self.trip_cut_off)
        self.copulas_we = return_copula(df=nts_df, weekday=2, year=self.year, trip_cut_off=self.trip_cut_off)

    def _fit_tools(self, tools_folder):

        self.get_num_journey_p_vector()
        self.gen_jour_seq_p_vector()
        self.gen_copulas()

        # Saving tools to tools folder

        tools_dict_path = tools_folder + "/tools.pkl"

        if Path(tools_dict_path).is_file():
            logging.info("File exists, Loading file...")

            with open(tools_dict_path, "rb") as file:  # 'rb' = read binary mode
                tools_dict = pickle.load(file)

            if self.year in tools_dict:
                logging.info("Tools for this year have already been saved!")

            else:
                logging.info(f"Saving tools for a new {self.year}")
                tools_dict[self.year] = {"num_journey_p_vector_wd": self.num_journey_p_vector_wd,
                "num_journey_p_vector_we": self.num_journey_p_vector_we,
                "jour_seq_p_vector_wd": self.jour_seq_p_vector_wd,
                "jour_seq_p_vector_we": self.jour_seq_p_vector_we,
                "copulas_wd": self.copulas_wd,
                "copulas_we": self.copulas_we
                }
            
                with open(tools_dict_path, "wb") as file:
                    pickle.dump(tools_dict, file)

                logging.info(f"Tools for {self.year} Saved!")

        else:
            logging.info("File does not exist...")
            logging.info("Creating file")

            tools_dict = {}
            tools_dict[self.year] = {"num_journey_p_vector_wd": self.num_journey_p_vector_wd,
                               "num_journey_p_vector_we": self.num_journey_p_vector_we,
                               "jour_seq_p_vector_wd": self.jour_seq_p_vector_wd,
                               "jour_seq_p_vector_we": self.jour_seq_p_vector_we,
                               "copulas_wd": self.copulas_wd,
                               "copulas_we": self.copulas_we
                               }
            
            with open(tools_dict_path, "wb") as file:
                pickle.dump(tools_dict, file)

            logging.info(f"Tools for {self.year} Saved!")

    def simulate(self):

        # Start Fresh

        self.mobility_schedule = pd.DataFrame()

        # Generate TS

        self._gen_ts()

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
    
# Quick loop to save all tools for every year

'''
for year in range(2005,2025):

    m = MobilitySimulator(nts_df=df, year=year, trip_cut_off=10)

    m.fit_tools(tools_folder=tools)

#m.simulate()


#print(m.mobility_schedule)
'''