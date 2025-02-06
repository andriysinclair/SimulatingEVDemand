import random



### Generate EVs

# Car types

type1 = {"Capacity (kWh)": 40,
 "Energy consumption (kWh/mi)": 0.3,
  "Market share": 0.3 }

type2 = {"Capacity (kWh)": 100,
 "Energy consumption (kWh/mi)": 0.3,
  "Market share": 0.6 }

type3 = {"Capacity (kWh)": 100,
 "Energy consumption (kWh/mi)": 0.35,
  "Market share": 0.1 }

car_types = [type1, type2, type3]

# Fu

class Agent:
    def __init__(self, car, trip_log, agent_id):
        agent_id = agent_id
        self.car = car
        self.trip_log = trip_log


    def generate_car(self, car_types = car_types):
        self.car = random.choices(car_types, [type_["Market share"] for type_ in car_types])

    def generate_first_soc(self):
        pass

    def generate_trip_log(self):
        pass

class ChargePoint:
    pass