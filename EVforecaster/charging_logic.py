import pandas as pd
import config as cfg
import random
import numpy as np
import logging

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO)

# Loading in df

full_df_path = cfg.root_folder + "/dataframes/day_trip_merge.pkl"
full_df = pd.read_pickle(full_df_path)


def generate_charger(x, home_charger_likelihood=0.96, work_charger_likelihood=0.62, public_charger_likelihood=0.17):
    if x == 3:
        return random.choices([1,0], weights=[home_charger_likelihood,1-home_charger_likelihood])[0]
    if x == 2:
        return random.choices([1,0], weights=[public_charger_likelihood,1-public_charger_likelihood])[0]
    else:
        return random.choices([1,0], weights=[work_charger_likelihood,1-work_charger_likelihood])[0]
    




def charging_logic(df, battery_size = cfg.battery_size, energy_efficiency=cfg.energy_efficiency,
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
                         "ChargeLoc": [],
                         "ChargingRate": []}
 
        for i in individual_ids[:test_index]:

            i_df = df[df["IndividualID"]==i]

            i_df = i_df.copy()

            # Setting initial SOC uniformly distributed

            first_trip_req_charge = (i_df.iloc[0]["TripDisExSW"]* energy_efficiency)/1000
            init_SOC = random.uniform(first_trip_req_charge, battery_size)

            # Calculating time at trip end location

            # bring trip start rolling forward
            i_df["TripStartRolling+1"] = i_df["TripStartRolling"].shift(-1)

            # Calculate time at end location
            i_df["TimeEndLoc"] = i_df["TripStartRolling+1"] - i_df["TripEndRolling"] 

            i_df["Distance+1"] = i_df["TripDisExSW"].shift(-1)
            i_df["Req_charge+1"] = (i_df["Distance+1"] * energy_efficiency)/1000

            # Working row wise to model charging decisions and change SOC
            SOC_list = [np.round( init_SOC, 2) ]
            charge_decision_list = []
            charge_start_time_list = []
            charge_end_time_list = []
            total_power_used_list = []

            for idx, row in i_df.iterrows():

                # For all rows but the last

                if idx != i_df.index[-1]:

                    # No charger available or insufficient stopping time

                    if row["IsCharger"] == 0 or row["TimeEndLoc"] < min_stop_time_to_charge:
                        
                        SOC_list.append(    np.round(  SOC_list[-1]-row["Req_charge+1"],  2))
                        charge_decision_list.append(0)
                        logging.debug("No charger or not enough parking time")
                        logging.debug(charge_decision_list)
                        logging.debug(SOC_list)
                    
                    else:
                        logging.debug("charger is available and car has stopped for sufficient time")
                        
                        SOC_percentage = SOC_list[-1]/battery_size
                        charge_decision_prob = SOC_charging_prob(SOC_percentage)
                        charge_decision = np.random.choice([0,1], p = [1-charge_decision_prob, charge_decision_prob])
                        charge_decision_list.append(charge_decision)

                        if charge_decision == 1:

                            charging_dict["IndividualID"].append(i)


                            location_charging_rate = charging_rates[row["TripEndLoc"]]
                            logging.debug(f"Location charging rate: {location_charging_rate}")

                            # If we have a charge...
                            logging.debug(f"Individual has chosen to charge")
                            total_possible_charge = (row["TimeEndLoc"]/60) * location_charging_rate
                            logging.debug(f"Total possible charge: {total_possible_charge}")

                            if total_possible_charge > (battery_size - SOC_list[-1]):
                                # Fully charge car
                                logging.debug(f"As total possible charge: {total_possible_charge} > battery size: {battery_size}")
                                logging.debug(f"Fully charging car")

                                total_power_used = battery_size - SOC_list[-1]

                                charging_dict["TotalPowerUsed"].append(total_power_used)

                                charge_start_time = row["TripEnd"]
                                charge_end_time = charge_start_time + total_power_used/location_charging_rate

                                # Rounding to nearest 5 minutes
                                charge_end_time = 5 * round(charge_end_time / 5)

                                charge_start_time_list.append(charge_start_time)
                                charge_end_time_list.append(charge_end_time)
                                total_power_used_list.append(total_power_used)

                                SOC_list.append(np.round(battery_size,2))

                                # Removing SOC from next trip
                                SOC_list[-1] = np.round(  SOC_list[-1]-row["Req_charge+1"],  2)

                                logging.debug(charge_decision_list)
                                logging.debug(SOC_list)


                            else:
                                
                                # total power used = total posible charge

                                charging_dict["TotalPowerUsed"].append(total_possible_charge)

                                charge_start_time = row["TripEnd"]
                                charge_end_time = charge_start_time + total_possible_charge/location_charging_rate

                                # Rounding to nearest 5 minutes
                                charge_end_time = 5 * round(charge_end_time / 5)

                                charge_start_time_list.append(charge_start_time)
                                charge_end_time_list.append(charge_end_time)
                                total_power_used_list.append(total_possible_charge)

                                SOC_list.append(np.round(   SOC_list[-1] + total_possible_charge, 2))

                                # Removing SOC from next trip
                                SOC_list[-1] = np.round(  SOC_list[-1]-row["Req_charge+1"],  2)

                                logging.debug("Applying maximum possible charge")
                                logging.debug(charge_decision_list)
                                logging.debug(SOC_list)

                            charging_dict["ChargeStart"].append(charge_start_time)
                            charging_dict["ChargeEnd"].append(charge_end_time)
                            charging_dict["ChargeLoc"].append(row["TripEndLoc"])
                            charging_dict["ChargingRate"].append(location_charging_rate)
                        
                        else:

                            # Removing SOC from next trip
                            SOC_list.append(  np.round(  SOC_list[-1]-row["Req_charge+1"],  2) )
                            logging.debug(f"Individual has chosen NOT to charge")
                            logging.debug(charge_decision_list)
                            logging.debug(SOC_list)



                else:
                    # For the final row
                    logging.debug(f"Final row so no next trip")

                    if row["IsCharger"] == 0:
                        logging.debug(f"No charger available")

                        charge_decision_list.append(0)
                        SOC_list.append(np.round( SOC_list[-1], 2))
                        logging.debug(charge_decision_list)
                        logging.debug(SOC_list)


                    else:

                        location_charging_rate = charging_rates[row["TripEndLoc"]]
                        logging.debug(f"Location charging rate: {location_charging_rate}")

                        logging.debug(f"Charger available")
                        SOC_percentage = SOC_list[-1]/battery_size
                        charge_decision_prob = SOC_charging_prob(SOC_percentage)
                        charge_decision = np.random.choice([0,1], p = [1-charge_decision_prob, charge_decision_prob])
                        charge_decision_list.append(charge_decision)

                        if charge_decision == 1:
                            logging.debug("Individual has chosen to charge")

                            charging_dict["IndividualID"].append(i)

                            total_power_used = battery_size - SOC_list[-1]
                            charge_start_time = row["TripEnd"]
                            charge_end_time = charge_start_time + total_power_used/location_charging_rate

                            # Rounding to nearest 5 minutes
                            charge_end_time = 5 * round(charge_end_time / 5)

                            charge_start_time_list.append(charge_start_time)
                            charge_end_time_list.append(charge_end_time)
                            total_power_used_list.append(total_power_used)

                            SOC_list.append(np.round(battery_size,2))
                            logging.debug(charge_decision_list)
                            logging.debug(SOC_list)

                            charging_dict["TotalPowerUsed"].append(total_power_used)
                            charging_dict["ChargeStart"].append(charge_start_time)
                            charging_dict["ChargeEnd"].append(charge_end_time)
                            charging_dict["ChargeLoc"].append(row["TripEndLoc"])
                            charging_dict["ChargingRate"].append(location_charging_rate)


                        else:
                            logging.debug("Individual has chosen NOT to charge")
                            SOC_list.append(np.round(   SOC_list[-1], 2))
                            logging.debug(charge_decision_list)
                            logging.debug(SOC_list)

            logging.info(SOC_list)
            logging.info(charge_decision_list)
            logging.info(total_power_used_list)
            logging.info(charge_start_time_list)
            logging.info(charge_end_time_list)


            logging.info(i_df[["TripEndLoc", "IsCharger", "TripDisExSW", "Distance+1", "Req_charge+1", "TripStartRolling", "TripEndRolling", "TripStartRolling+1", "TimeEndLoc"]].head())
            print("")

        charging_df = pd.DataFrame(charging_dict)

        return charging_df








        
if __name__ == "__main__":

    test = charging_logic(full_df, test_index=5)

    test.to_csv(cfg.root_folder + "/output_csvs/charging_df_2017.csv", index=False)
