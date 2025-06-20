import pandas as pd
import config as cfg
import random
import numpy as np
import logging
import config as cfg

def generate_charger(x, home_charger_likelihood, work_charger_likelihood, public_charger_likelihood) -> int:
    """
    generate_charger 

    Generates a charger or no charger for a location with a given probability as defined in config.py 

    Args:
        x (int): location [1,2,3]
        home_charger_likelihood (int, optional): probability [0,1]. Defaults to cfg.charger_likelihood["home"].
        work_charger_likelihood (int, optional): prob [0,1]. Defaults to cfg.charger_likelihood["work"].
        public_charger_likelihood (int, optional): prob [0,1]. Defaults to cfg.charger_likelihood["other"].

    Returns:
        int: A 1 or a 0 as obtained through probability
    """    
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
                       SOC_charging_prob,
                       car_type) -> int:
    """
    obtain_decision_to_charge 

    Obtains a yes (1) or no (0) decision to charge based on SOC and other requirmenets such as minimum stop time

    Args:
        SOC (float): State of Charge (kW)
        available_charger (int): available (1) not available (0)
        time_duration_at_location (float): time at location
        last_trip_flag (bool): is last trip?
        min_stop_time_to_charge (float): in minutes
        battery_size (float): in kW
        SOC_charging_prob (func): function to determine charge probability based off SOC, available in config.py
        car_type (str): BEV or PHEV, changes decision function

    Returns:
        int: 1 (charge) or 0 (not charge)
    """    
    
    if available_charger == 0:

        logging.debug(f"No charger available")
        return 0, 0
    
    
    elif time_duration_at_location < min_stop_time_to_charge and not last_trip_flag:

        logging.debug(f"Insufficient time spent at location to charge")
        return 0, 0
    
    logging.debug("charger is available and car has stopped for sufficient time")
    SOC_percentage = SOC/battery_size

    # Function calculates charging probabilites based off SOC

    charge_decision_prob = SOC_charging_prob(SOC_percentage, car_type=car_type)

    logging.debug(f"charging decision prob: {charge_decision_prob}")

    # Draws 1,0 from charging decision probability
    charge_decision = np.random.choice([0,1], p = [1-charge_decision_prob, charge_decision_prob])

    logging.debug(f"charging decision: {charge_decision}")

    return int(charge_decision), charge_decision_prob

def calculate_charging_session(SOC, location_charging_rate, time_duration_at_location, last_trip_flag,
                               charge_start_time, battery_size, home_shift)->tuple:
    """
    calculate_charging_session 

    Creates a charging session if individual decides to charge.
    Obtains charge extracted and the end of charging session.

    Args:
        SOC (float): State of charge [0, battery size]
        location_charging_rate (float): in kWh
        time_duration_at_location (float): in minutes
        last_trip_flag (bool): True if this is individuals last trip of the week
        charge_start_time (float): between 0 and 1440 (24hrs)
        battery_size (float): kW

    Returns:
        tuple: returns 
                - float, new SOC following charge
                - float, energy obtained from charge (kW)
                - float, charge start time [0,1440]
                - float, charge end time [0,1440]
                - float, charge duration (minutes)

    """    
    
    # Assuming individuals charge when they arrive at end location - this is charge start time
    
    charge_start_time = 5 * round(charge_start_time/5) + home_shift

    # Capacity at charging location
    remaining_capacity = battery_size - SOC
    logging.debug(f"Remaining capacity: {remaining_capacity:.2f}")

    # If this is last trip there is no next trip for which to charge. Therefore if charging is chosen it will automatically
    # result in a full charge

    if last_trip_flag or pd.isna(time_duration_at_location):
        logging.debug("Last trip or unknown duration — assume full charge")
        charge_energy = remaining_capacity

    else:
        # If not last trip individual charges the minimum of a total possible charge given 
        # time at location or their remaining capacity
        total_possible_charge = (time_duration_at_location / 60) * location_charging_rate
        logging.debug(f"Total possible charge: {total_possible_charge:.2f}")
        charge_energy = min(total_possible_charge, remaining_capacity)

    # Adding the new charge to the SOC
    new_SOC = SOC + charge_energy

    # Obtaining charge duration and charge end time
    charge_duration = (charge_energy / location_charging_rate) * 60  # in minutes
    charge_duration = 5 * round(charge_duration / 5)
    charge_end_time = charge_start_time + charge_duration

    logging.debug(f"New SOC: {new_SOC}")
    logging.debug(f"charge energy: {charge_energy}")
    logging.debug(f"charge start time: {charge_start_time}")
    logging.debug(f"charge end time: {charge_end_time}")
    logging.debug(f"charge duration: {charge_duration}")   

    return new_SOC, charge_energy, charge_start_time, charge_end_time, charge_duration