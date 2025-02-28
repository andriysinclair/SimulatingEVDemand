import pandas as pd
import logging
from pathlib import Path
from Auxillary_functions import return_num_journey_prob, return_journey_seq, return_copula, gen_cont_seq
import random
import pickle
import concurrent.futures

from MobilitySimulatorDev import MobilitySimulator

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', force=True)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Tools folder
tools = str(Path(__file__).resolve().parent / "Tools")

#Agents folder
agents = str(Path(__file__).resolve().parent / "Agents")

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


if __name__ == "__main__":

    ms = MOBSIM(2017)

    i_s, calc_df, sim_times = ms.simulate(1,1,1,7,5, track_performance=True)

    # Save to a file
    with open(agents + "/agents.pkl", "wb") as f:
        pickle.dump(i_s, f)

    print("Agents saved successfully!")

    # Save to a file
    with open(agents + "/calculations.pkl", "wb") as f:
        pickle.dump(calc_df, f)

    print("Calculation df saved successfully!")

    # Save to a file
    with open(agents + "/sim_times.pkl", "wb") as f:
        pickle.dump(calc_df, f)

    print("Sim times saved successfully!")
    