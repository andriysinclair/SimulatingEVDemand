import pandas as pd
import config as cfg
import random
import numpy as np
import logging

def generate_charger(x, home_charger_likelihood=0.96, work_charger_likelihood=0.62, public_charger_likelihood=0.17):
    if x == 3:
        return random.choices([1,0], weights=[home_charger_likelihood,1-home_charger_likelihood])[0]
    if x == 2:
        return random.choices([1,0], weights=[public_charger_likelihood,1-public_charger_likelihood])[0]
    else:
        return random.choices([1,0], weights=[work_charger_likelihood,1-work_charger_likelihood])[0]

### Auxillary functions for charging logic ###

def obtain_decision_to_charge(SOC, available_charger, time_duration_at_location, last_trip_flag,
                       min_stop_time_to_charge,
                       battery_size,
                       SOC_charging_prob):
    
    if available_charger == 0:

        logging.debug(f"No charger available")
        return 0
    
    
    elif time_duration_at_location < min_stop_time_to_charge and not last_trip_flag:

        logging.debug(f"Insufficient time spent at location to charge")
        return 0
    
    logging.debug("charger is available and car has stopped for sufficient time")
    SOC_percentage = SOC/battery_size
    charge_decision_prob = SOC_charging_prob(SOC_percentage)

    logging.debug(f"charging decision prob: {charge_decision_prob}")

    charge_decision = np.random.choice([0,1], p = [1-charge_decision_prob, charge_decision_prob])

    logging.debug(f"charging decision: {charge_decision}")

    return int(charge_decision)

def calculate_charging_session(SOC, location_charging_rate, time_duration_at_location, last_trip_flag,
                               charge_start_time, battery_size):
    
    charge_start_time = 5 * round(charge_start_time/5)

    remaining_capacity = battery_size - SOC
    logging.debug(f"Remaining capacity: {remaining_capacity:.2f}")

    if last_trip_flag or pd.isna(time_duration_at_location):
        logging.debug("Last trip or unknown duration — assume full charge")
        charge_energy = remaining_capacity
    else:
        total_possible_charge = (time_duration_at_location / 60) * location_charging_rate
        logging.debug(f"Total possible charge: {total_possible_charge:.2f}")
        charge_energy = min(total_possible_charge, remaining_capacity)

    new_SOC = SOC + charge_energy
    charge_duration = (charge_energy / location_charging_rate) * 60  # in minutes
    charge_duration = 5 * round(charge_duration / 5)

    charge_end_time = charge_start_time + charge_duration

    logging.debug(f"New SOC: {new_SOC}")
    logging.debug(f"charge energy: {charge_energy}")
    logging.debug(f"charge start time: {charge_start_time}")
    logging.debug(f"charge end time: {charge_end_time}")
    logging.debug(f"charge duration: {charge_duration}")   

    return new_SOC, charge_energy, charge_start_time, charge_end_time, charge_duration