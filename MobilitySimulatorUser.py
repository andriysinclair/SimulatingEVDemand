import pandas as pd
import logging
from pathlib import Path
from Auxillary_functions import return_num_journey_prob, return_journey_seq, return_copula, gen_cont_seq
import random
import pickle
import concurrent.futures

from MobilitySimulatorDev import MobilitySimulator

# Configure basic logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Tools folder
tools = str(Path(__file__).resolve().parent / "Tools")

class MOBSIM(MobilitySimulator):

    def __init__(self, year):
        super().__init__(year)
        self._set_params(year=year)

    def _set_params(self, year, tools_folder=tools):

        self.year = year

        tools_dict_path = tools_folder + "/tools.pkl"

        if year not in range(2005, 2025):
            raise ValueError(f"Year must be in range 2005 - 2024, year is {year}.")
        
        else:
            logging.debug(f"Initialising tools for {year}")

            with open(tools_dict_path, "rb") as file:  # 'rb' = read binary mode
                tools_dict = pickle.load(file)

            self.num_journey_p_vector_we = tools_dict[year]["num_journey_p_vector_we"]
            self.num_journey_p_vector_wd = tools_dict[year]["num_journey_p_vector_wd"]

            self.jour_seq_p_vector_we = tools_dict[year]["jour_seq_p_vector_we"]
            self.jour_seq_p_vector_wd = tools_dict[year]["jour_seq_p_vector_wd"]

            self.copulas_we = tools_dict[year]["copulas_we"]
            self.copulas_wd = tools_dict[year]["copulas_wd"]


        



