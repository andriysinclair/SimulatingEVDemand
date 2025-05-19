import os
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from pathlib import Path

# Absolute Paths

root_folder = str(Path(__file__).parent.parent)

trip_cols_to_keep = [
    "TripID",
    "DayID",
    "IndividualID",
    "HouseholdID",
    "PSUID",
    "PersNo",
    "TravDay",
    "JourSeq",
    "TripStart",
    "TripEnd",
    "TripDisExSW",
    "TripOrigGOR_B02ID",
    "TripDestGOR_B02ID",
    "TripPurpFrom_B01ID",
    "TripPurpTo_B01ID"
]

day_cols_to_keep = [
    "DayID",
    "TravelYear",
    "TravelWeekDay_B01ID"
]

# Work: 1
# Other: 2
# Home: 3

trip_purpouse_mapping = {
    1: 1,   # Work
    2: 1,   # In course of work
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


if __name__ == "__main__":
    print(root_folder)
