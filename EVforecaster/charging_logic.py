import pandas as pd
import config as cfg
import random
import numpy as np
import pickle
import logging
from charging_logic_auxillary import generate_charger, obtain_decision_to_charge, calculate_charging_session




def charging_logic(df, travel_weeks = list(range(1,55)), output_file_name=None,  
                   battery_size_phev = cfg.battery_size_phev,
                   battery_size_bev = cfg.battery_size_bev,
                   car_types = cfg.car_types,
                    is_loaded=False, test_index=None,
                    charging_rates = cfg.charging_rates,
                    save_schedule = False
                    ) -> pd.DataFrame:
    """
    charging_logic 

    Takes a transformed travel df and paramaters on charging rates, battery capacities etc. from config.py
    and uses them to create simulated charging schedules   

    Args:
        df (pd.DataFrame): Transformed travel DF
        output_file_name (str): Named of saved file. Saves as pickle to dataframes folder and as csv to output_csvs folder
        battery_size (int, optional): Battery Size (KW). Defaults to cfg.battery_size.
        energy_efficiency (int, optional): Energy efficiency (KW/mile). Defaults to cfg.energy_efficiency.
        is_loaded (bool, optional): True if charging schedules have been generated, then simply loades from /dataframes. Defaults to False.
        test_index (int, optional): Number of individuals on which to test (used for debugging). Defaults to None.
        min_stop_time_to_charge (int, optional): Minimum amount of time individual needs to stop in order to consider charging. Defaults to cfg.min_stop_time_to_charge.
        SOC_charging_prob (func, optional): Probabilistic function to determine charging likelihood based off SOC. Defaults to cfg.SOC_charging_prob.
        charging_rates (int, optional): Charging Rate (kWh). Defaults to cfg.charging_rates.

    Returns:
        pd.DataFrame: Charging schedules appended to transformed travel df/
    """    

    # Adding nodes with available chargers

    df = df.copy()

    # Filter by travel week

    df = df[df["TWSWeekNew"].isin(travel_weeks)]

    # Set chargers given the location and probability of charger existing at that location
    df["IsCharger"] = df["TripEndLoc"].apply(lambda x: generate_charger(x))

    # Obtain battery size

    if not is_loaded:

        # Looping over individuals
        individual_ids = df["IndividualID"].unique()

        # Adding new variables based on charge
        charging_dict = {"TripID": [],
                         "TotalPowerUsed": [],
                         "ChargeStart": [],
                         "ChargeEnd": [],
                         "ChargeDuration": [],
                         "ChargeLoc": [],
                         "ChargingRate": [],
                         "TravelDay": [],
                         "TravelWeek": [],
                         "TravelYear": [],
                         "CarType": [],
                         "BatterySize": [],
                         "Efficiency": []} 
        
        total_trips = 0
        negative_trips = 0
 
        for i in individual_ids[:test_index]:

            # Draw the individual's battery size and EV type: BEV or PHEV

            i_df = df[df["IndividualID"]==i]
            i_df = i_df.copy()

            # Obtaining a car type for an individual
            car_type_for_i = np.random.choice(car_types[0], p=car_types[1])

            i_df["CarType"] = car_type_for_i

            # Obtaining battery type and efficiency for an individual
            if car_type_for_i == "PHEV":

                batt_efficiency_tupe = random.choices(battery_size_phev[0], battery_size_phev[1])[0]

                battery_size_for_i = batt_efficiency_tupe[0]
                i_df["BatterySize"] = battery_size_for_i

                efficiency_for_i = batt_efficiency_tupe[1]
                i_df["Efficiency"] = efficiency_for_i

            else:

                batt_efficiency_tupe = random.choices(battery_size_bev[0], battery_size_bev[1])[0]
                battery_size_for_i = batt_efficiency_tupe[0]
                i_df["BatterySize"] = battery_size_for_i

                efficiency_for_i = batt_efficiency_tupe[1]
                i_df["Efficiency"] = efficiency_for_i


            # Obtaining the required charge for first trip
            first_trip_req_charge = (i_df.iloc[0]["TripDisExSW"]* efficiency_for_i)/1000

            # Setting initital SOC with uniformally with the following bounds
            init_SOC = random.uniform(first_trip_req_charge, battery_size_for_i)
            logging.debug(f"Intitial SOC: {init_SOC}")

            # Calculating energy required for the next trip

            i_df["Req_charge+1"] = (i_df["Distance+1"] * efficiency_for_i)/1000

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
                trip_id = row["TripID"]
                available_charger = row["IsCharger"]
                end_location = row["TripEndLoc"]
                #charging_rate = charging_rates[row["TripEndLoc"]]   
                charging_rate = np.random.choice(   charging_rates[row["TripEndLoc"]][0], p=  charging_rates[row["TripEndLoc"]][1] )
                #logging.info(f"Selected charging rate: {charging_rate}")
                time_duration_at_location = row["TimeEndLoc"]
                time_at_location = row["TripEnd"]
                current_SOC = SOC_list[-1]
                charge_for_next_trip = row["Req_charge+1"]
                travel_day = row["TravelWeekDay_B01ID"]
                travel_week = row["TWSWeekNew"]
                travel_year = row["TravelYear"]

                # If current SOC is negative and we have a BEV consider faulty trip and we move on to the next person
                if current_SOC < 0 and car_type_for_i == "BEV":
                    negative_trips += 1
                    break

                if current_SOC < 0  and car_type_for_i == "PHEV":
                    current_SOC = 0

                # Bool to inform us if it is the last trip
                if idx == i_df.index[-1]:
                    last_trip_flag = True
                else:
                    last_trip_flag = False
                
                
                logging.debug(f"Current SOC: {current_SOC}")
                logging.debug(f"charge required for next trip: {charge_for_next_trip}")


                total_trips += 1
                
                # Obtain individuals decision to charge

                decision_to_charge = obtain_decision_to_charge(SOC=current_SOC, available_charger=available_charger,
                                                               time_duration_at_location=time_duration_at_location,
                                                               last_trip_flag=last_trip_flag,
                                                               min_stop_time_to_charge=cfg.min_stop_time_to_charge,
                                                               battery_size=battery_size_for_i,
                                                               SOC_charging_prob=cfg.SOC_charging_prob,
                                                               car_type=car_type_for_i)
                
                charge_decision_list.append(decision_to_charge)

                if decision_to_charge == 1:

                    try:
                    
                        new_SOC, total_power_used, charge_start_time, charge_end_time, charge_duration = calculate_charging_session(SOC=current_SOC, location_charging_rate=charging_rate,
                                                                                                                                time_duration_at_location=time_duration_at_location,
                                                                                                                                charge_start_time=time_at_location, last_trip_flag=last_trip_flag,
                                                                                                                                battery_size=battery_size_for_i)
                    
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
                    #charging_dict["IndividualID"].append(i)
                    charging_dict["TripID"].append(trip_id)
                    charging_dict["TravelYear"].append(travel_year)
                    charging_dict["TravelWeek"].append(travel_week)
                    charging_dict["TravelDay"].append(travel_day)
                    charging_dict["TotalPowerUsed"].append(total_power_used)
                    charging_dict["ChargeStart"].append(charge_start_time)
                    charging_dict["ChargeEnd"].append(charge_end_time)
                    charging_dict["ChargeDuration"].append(charge_duration)
                    charging_dict["ChargingRate"].append(charging_rate)
                    charging_dict["ChargeLoc"].append(end_location)
                    charging_dict["CarType"].append(car_type_for_i)
                    charging_dict["BatterySize"].append(battery_size_for_i)
                    charging_dict["Efficiency"].append(efficiency_for_i)

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

            logging.debug(i)
            logging.debug(car_type_for_i)
            logging.debug(charge_decision_list)
            logging.debug(SOC_list)
            

        charging_df = pd.DataFrame(charging_dict)

        charging_df = pd.merge(charging_df, df, on="TripID")
        logging.info(f"Total trips: {total_trips}")
        logging.info(f"Negative trips: {negative_trips}")
        logging.info(f"% negative trips: {negative_trips/total_trips*100:.2f}%")


        # Dump charging schedules to pickle
        if save_schedule:
            with open(cfg.root_folder + f"/dataframes/{output_file_name}.pkl", "wb") as f:
                pickle.dump(charging_df, f)   

            charging_df.to_csv(cfg.root_folder + f"/output_csvs/{output_file_name}.csv", index=False)

            logging.info(f"File saved to {output_file_name}")


        return charging_df

        
if __name__ == "__main__":

    # Set up basic configuration for logging
    logging.basicConfig(level=logging.INFO)

    # Loading in df

    full_df_path = cfg.root_folder + "/dataframes/Ready_to_model_df_[2017].pkl"
    full_df = pd.read_pickle(full_df_path)

    test = charging_logic(full_df, output_file_name="charging_df", test_index=None, save_schedule=True)

    
