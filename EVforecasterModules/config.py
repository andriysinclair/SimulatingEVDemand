import os
from pathlib import Path
from scipy.stats import norm
import numpy as np

# Absolute Paths

root_folder = str(Path(__file__).parent.parent)

trip_cols_to_keep = [
    "DayID",
    "TripID",
    "IndividualID",
    "HouseholdID",
    "TravDay",
    "JourSeq",
    "TripStart",
    "TripEnd",
    "TripDisExSW",
    #"TripOrigGOR_B02ID",
    #"TripDestGOR_B02ID",
    "TripPurpFrom_B01ID",
    "TripPurpTo_B01ID",
    #"SurveyYear"
    
]

day_cols_to_keep = [
    "DayID",
    "TravelYear",
    "TravelWeekDay_B01ID"
]

household_cols_to_keep = [
    "TWSWeek",   # Travel Week Start - Week number in calendar year
    "HouseholdID"
]



# Work: 1
# Other: 2
# Home: 3

trip_purpouse_mapping = {
    1: 1,   # Work
    2: 2,   # In course of work
    3: 2,   # Education
    4: 2,   # Food shopping
    5: 2,   # Non food shopping
    6: 2,   # Personal business medical
    7: 2,   # Personal business eat / drink
    8: 2,   # Personal business other
    9: 2,   # Eat / drink with friends
    10: 2,  # Visit friends
    11: 2,  # Other social
    12: 2,  # Entertain /  public activity
    13: 2,  # Sport: participate
    14: 2,  # Holiday: base
    15: 2,  # Day trip / just walk
    16: 2,  # Other non-escort
    17: 2,  # Escort home
    18: 2,  # Escort work
    19: 2,  # Escort in course of work
    20: 2,  # Escort education
    21: 2,  # Escort shopping / personal business
    22: 2,  # Other escort
    23: 3}   # Home

trip_type_mapping = {

    "3-1": 1,   #Home-Work
    "1-3": 2,   #Work-Home

    "3-2": 3,   #Home-Other
    "2-3": 4,   #Other-Home

    "1-2": 5,   #Work-Other
    "2-1": 6,   #Other-Work

    "1-1": 7,   #Home-Home
    "2-2": 8,   #Other-Other
    "3-3": 9,   #Work-Work
}


### Charging logic configuration ###

car_types = [  ["PHEV", "BEV"], [0.7, 0.3]  ]
battery_size_phev = [ [  (13.8, 363 )    ], [1] ]
battery_size_bev = [ [   (25,269)   ], [1]  ]


min_stop_time_to_charge = 120

home_shift = 60

                  
charging_rates = {1: [[3.6], [1]],            # Work
                  2: [[11], [1]],            # Other
                  3: [[3, 7], [0.5, 0.5] ]}            # Home

charger_likelihood = {"work": 0,
                      "other": 0,
                      "home": 1}

def SOC_charging_prob(soc, car_type, mu=0.6, sigma=0.2, ):
    cdf = norm.cdf(soc, loc=mu, scale=sigma)
    prob = 1 - cdf

    # Ensure P(charge) = 1 if SOC=0
    if car_type=="BEV":
        prob = np.where(soc<=0, 1.0, prob)

    return prob

# SOC-based charging decision model

if __name__ == "__main__":
    print(root_folder)
