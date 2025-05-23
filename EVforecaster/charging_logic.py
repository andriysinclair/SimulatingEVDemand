import pandas as pd
import config as cfg
import random
import numpy as np
import pickle
import logging
from charging_logic_auxillary import generate_charger, obtain_decision_to_charge, calculate_charging_session

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO)

# Loading in df

full_df_path = cfg.root_folder + "/dataframes/Ready_to_model_df.pkl"
full_df = pd.read_pickle(full_df_path)


def charging_logic(df, output_file_name,  battery_size = cfg.battery_size, energy_efficiency=cfg.energy_efficiency,
                    is_loaded=False, test_index=None,
                    min_stop_time_to_charge = cfg.min_stop_time_to_charge,
                    SOC_charging_prob = cfg.SOC_charging_prob,
                    charging_rates = cfg.charging_rates):

    # Adding nodes with available chargers

    df = df.copy()

    df["IsCharger"] = df["TripEndLoc"].apply(lambda x: generate_charger(x))

    if not is_loaded:

        individual_ids = df["IndividualID"].unique()

        charging_dict = {"IndividualID": [],
                         "TotalPowerUsed": [],
                         "ChargeStart": [],
                         "ChargeEnd": [],
                         "ChargeDuration": [],
                         "ChargeLoc": [],
                         "ChargingRate": [],
                         "TravelDay": [],
                         "TravelWeek": [],
                         "TravelYear": []}
        
        total_trips = 0
        negative_trips = 0
 
        for i in individual_ids[:test_index]:

            i_df = df[df["IndividualID"]==i]
            i_df = i_df.copy()

            # Setting initial SOC uniformly distributed

            first_trip_req_charge = (i_df.iloc[0]["TripDisExSW"]* energy_efficiency)/1000
            init_SOC = random.uniform(first_trip_req_charge, battery_size)
            logging.debug(f"Intitial SOC: {init_SOC}")

            # Calculating time at trip end location

            i_df["Req_charge+1"] = (i_df["Distance+1"] * energy_efficiency)/1000

            #logging.debug(i_df[["TravelWeekDay_B01ID", "WeekDayDiff", "WeekRollover", "TWSWeek", "TWSWeekNew"]])

            # Working row wise to model charging decisions and change SOC
            SOC_list = [np.round( init_SOC, 2) ]
            charge_decision_list = []
            charge_start_time_list = []
            charge_end_time_list = []
            charge_duration_list = []
            total_power_used_list = []

            for idx, row in i_df.iterrows():

                # Obtain all relevant parameters from that trip
                available_charger = row["IsCharger"]
                end_location = row["TripEndLoc"]
                charging_rate = charging_rates[row["TripEndLoc"]]
                time_duration_at_location = row["TimeEndLoc"]
                time_at_location = row["TripEnd"]
                current_SOC = SOC_list[-1]
                charge_for_next_trip = row["Req_charge+1"]
                travel_day = row["TravelWeekDay_B01ID"]
                travel_week = row["TWSWeekNew"]
                travel_year = row["TravelYear"]

                # If current SOC is negative we move onto next person
                if current_SOC < 0:
                    break

                if idx == i_df.index[-1]:
                    last_trip_flag = True
                else:
                    last_trip_flag = False
                
                
                logging.debug(f"Current SOC: {current_SOC}")
                logging.debug(f"charge required for next trip: {charge_for_next_trip}")

                total_trips += 1
                
                if current_SOC < 0:
                    negative_trips += 1

                # Obtain individuals decision to charge

                decision_to_charge = obtain_decision_to_charge(SOC=current_SOC, available_charger=available_charger,
                                                               time_duration_at_location=time_duration_at_location,
                                                               last_trip_flag=last_trip_flag,
                                                               min_stop_time_to_charge=cfg.min_stop_time_to_charge,
                                                               battery_size=cfg.battery_size,
                                                               SOC_charging_prob=cfg.SOC_charging_prob)
                
                charge_decision_list.append(decision_to_charge)

                if decision_to_charge == 1:

                    try:
                    
                        new_SOC, total_power_used, charge_start_time, charge_end_time, charge_duration = calculate_charging_session(SOC=current_SOC, location_charging_rate=charging_rate,
                                                                                                                                time_duration_at_location=time_duration_at_location,
                                                                                                                                charge_start_time=time_at_location, last_trip_flag=last_trip_flag,
                                                                                                                                battery_size=cfg.battery_size)
                    
                    except Exception:

                        logging.info(f"Calculation failed! for individual: {i}")
                        logging.info(f"Charge details...")
                        logging.info(f"current SOC: {current_SOC}")
                        logging.info(f"Time duration at location: {time_duration_at_location}")
                        logging.info(f"charge start time: {time_at_location}")
                        logging.info(f"Last trip flag: {last_trip_flag}")


                        logging.info(f"SOC list:               {[int(i) for i in SOC_list]}")
                        logging.info(f"Charge decisions:       {[int(i) for i in charge_decision_list]}")
                        logging.info(f"Charge start times:     {[int(i) for i in charge_start_time_list]}")
                        logging.info(f"Charge end times:       {[int(i) for i in charge_end_time_list]}")
                        logging.info(f"Charge durations:       {[int(i) for i in charge_duration_list]}")
                        logging.info(f"Total power used (kWh): {[int(i) for i in total_power_used_list]}")
                        print("")



                    # Populate dictionary
                    charging_dict["IndividualID"].append(i)
                    charging_dict["TravelYear"].append(travel_year)
                    charging_dict["TravelWeek"].append(travel_week)
                    charging_dict["TravelDay"].append(travel_day)
                    charging_dict["TotalPowerUsed"].append(total_power_used)
                    charging_dict["ChargeStart"].append(charge_start_time)
                    charging_dict["ChargeEnd"].append(charge_end_time)
                    charging_dict["ChargeDuration"].append(charge_duration)
                    charging_dict["ChargingRate"].append(charging_rate)
                    charging_dict["ChargeLoc"].append(end_location)

                    # Removing charge required for next trip
                    if not last_trip_flag:
                        new_SOC -= charge_for_next_trip

                    charge_start_time_list.append(charge_start_time)
                    charge_end_time_list.append(charge_end_time)
                    charge_duration_list.append(charge_duration)
                    total_power_used_list.append(total_power_used)

                else:

                    if not last_trip_flag:
                        new_SOC = current_SOC -  charge_for_next_trip
                    
                    

                # Removing next trip from SOC
                
                SOC_list.append(new_SOC)

                logging.debug("")

            logging.debug(charge_decision_list)
            logging.debug(SOC_list)
            

        charging_df = pd.DataFrame(charging_dict)
        logging.info(f"Total trips: {total_trips}")
        logging.info(f"Negative trips: {negative_trips}")
        logging.info(f"% negative trips: {negative_trips/total_trips*100:.2f}%")

        # Dump charging schedules to pickle
        with open(cfg.root_folder + f"/dataframes/{output_file_name}", "wb") as f:
            pickle.dump(charging_df, f)   

        logging.info(f"File saved to {output_file_name}")


        return charging_df

        
if __name__ == "__main__":

    test = charging_logic(full_df, output_file_name="charging_df.pkl", test_index=None)

    test.to_csv(cfg.root_folder + "/output_csvs/charging_df_2017.csv", index=False)
