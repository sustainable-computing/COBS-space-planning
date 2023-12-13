import numpy as np
from cobs.model import Model
import datetime
import sys
import utility
import tqdm
import csv

sys.path.insert(0, utility.file_path)
Model.set_energyplus_folder(utility.eplus_location)


def flatten_dict(data: dict) -> dict:
    """
    The flatten_dict function takes a dictionary as input and returns a new dictionary with all the keys flattened.
    For example, if the input is:
    {'a': 1, 'b': {'x': 2, 'y': 3}, 'c': 4}
    the output will be:
    {'a': 1, 'b x': 2, 'b y', 3,'c', 4}

    :param data: dict: Pass in the dictionary to be flattened
    :return: A dictionary with the keys flattened
    """

    result = dict()
    for k in data:
        if isinstance(data[k], dict):
            for sub_key in data[k]:
                result[f"{k} {sub_key}"] = data[k][sub_key]
        else:
            result[k] = data[k]
    return result


def generate_mdd() -> None:
    """
    The generate_mdd function is used to generate a model description dictionary (MDD) file for the EnergyPlus
    model. The MDD contains information about the state and action spaces, as well as other metadata that can be
    used by an agent to interact with the environment.

    :return: None
    """

    # Define all control zones in the building model
    available_zones = ['TopFloor_Plenum', 'MidFloor_Plenum', 'FirstFloor_Plenum',
                       'Core_top', 'Core_mid', 'Core_bottom',
                       'Perimeter_top_ZN_3', 'Perimeter_top_ZN_2', 'Perimeter_top_ZN_1', 'Perimeter_top_ZN_4',
                       'Perimeter_bot_ZN_3', 'Perimeter_bot_ZN_2', 'Perimeter_bot_ZN_1', 'Perimeter_bot_ZN_4',
                       'Perimeter_mid_ZN_3', 'Perimeter_mid_ZN_2', 'Perimeter_mid_ZN_1', 'Perimeter_mid_ZN_4']
    control_zones = available_zones[3:]
    airloops = {'Core_top': "PACU_VAV_top", 'Core_mid': "PACU_VAV_mid", 'Core_bottom': "PACU_VAV_bot",
                'Perimeter_top_ZN_3': "PACU_VAV_top", 'Perimeter_top_ZN_2': "PACU_VAV_top",
                'Perimeter_top_ZN_1': "PACU_VAV_top", 'Perimeter_top_ZN_4': "PACU_VAV_top",
                'Perimeter_bot_ZN_3': "PACU_VAV_bot", 'Perimeter_bot_ZN_2': "PACU_VAV_bot",
                'Perimeter_bot_ZN_1': "PACU_VAV_bot", 'Perimeter_bot_ZN_4': "PACU_VAV_bot",
                'Perimeter_mid_ZN_3': "PACU_VAV_mid", 'Perimeter_mid_ZN_2': "PACU_VAV_mid",
                'Perimeter_mid_ZN_1': "PACU_VAV_mid", 'Perimeter_mid_ZN_4': "PACU_VAV_mid"}

    # Add state variables that we care about
    eplus_extra_states = {("Zone Air Relative Humidity", zone): f"{zone} humidity" for zone in available_zones}
    eplus_extra_states.update(
        {("Heating Coil Electric Energy", f"{zone} VAV Box Reheat Coil"):
             f"{zone} vav energy" for zone in available_zones})
    eplus_extra_states.update(
        {("Air System Electric Energy", airloop):
             f"{airloop} energy" for airloop in set(airloops.values())})
    eplus_extra_states[('Site Outdoor Air Drybulb Temperature', 'Environment')] = "outdoor temperature"
    eplus_extra_states[('Site Direct Solar Radiation Rate per Area', 'Environment')] = "site solar radiation"
    eplus_extra_states[('Facility Total HVAC Electric Demand Power', 'Whole Building')] = "total hvac"
    eplus_extra_states[('Schedule Value', 'Hours_of_operation')] = "operating"

    eplus_actuators_log = dict()
    with open("log_actuators.edd", "r") as edd_file:
        for line in edd_file:
            line = line.strip()
            if not line:
                continue
            _, actuator_key, component_type, control_type, _ = line.split(',')
            eplus_actuators_log[(component_type, control_type, actuator_key)] = \
                f"{component_type},{control_type},{actuator_key}"

    model = Model(idf_file_name="./eplus_files/OfficeMedium_Denver.idf",
                  weather_file="./eplus_files/USA_CO_Denver-Aurora-Buckley.AFB_.724695_TMY3.epw",
                  eplus_naming_dict=eplus_extra_states,
                  eplus_get_actuators=eplus_actuators_log)
    model.set_runperiod(365, start_month=1)
    model.set_timestep(4)

    # Define the control values
    for people_object in model.get_configuration("People"):
        model.add_configuration("Schedule:Constant",
                                {"Name": f"{people_object['Zone_or_ZoneList_Name']} Customized Occupancy Schedule",
                                 "Schedule Type Limits Name": "Any Number",
                                 "Hourly Value": 0})

        model.edit_configuration("People",
                                 {"Name": people_object["Name"]},
                                 {"Number of People Calculation Method": "People",
                                  "Number of People": 1,
                                  "Number of People Schedule Name":
                                      f"{people_object['Zone_or_ZoneList_Name']} Customized Occupancy Schedule"})

    for key, _ in eplus_extra_states.items():
        model.add_configuration("Output:Variable",
                                {"Key Value": key[1], "Variable Name": key[0], "Reporting Frequency": "Timestep"})

    # Assume all occupants are evenly distributed
    state = model.reset()
    occupants = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6]
    with tqdm.tqdm(total=4 * 24 * 365) as pbar:
        while not model.is_terminate():
            actions = list()

            next_time = state['time'] + datetime.timedelta(minutes=15)
            for i, zone in enumerate(control_zones):
                occupant = 0
                if state['operating'] and 9 <= next_time.hour < 17:
                    occupant = occupants[i]

                actions.append({"priority": 0,
                                "component_type": "Schedule:Constant",
                                "control_type": "Schedule Value",
                                "actuator_key": f"{zone} Customized Occupancy Schedule",
                                "value": occupant,
                                "start_time": state['timestep']})

            pbar.update(1)
            state = model.step(actions)


def run_fixed_occupancy(occupants: list[int], logfile_name: str, random_occupants: bool) -> None:
    """
    The run_fixed_occupancy function is used to run the EnergyPlus model with a given fixed occupancy schedule.
    The occupancy schedule is specified by the occupants list, which contains an integer for each zone in the building.
    The function will then set up and run EnergyPlus using this occupancy schedule, and write out all of its results to
    a CSV file.

    :param occupants: list[int]: Specify the number of occupants in each zone
    :param logfile_name: str: Specify the name of the log file
    :param random_occupants: bool: Specify whether the occupancy schedule should be random or not
    :return: Nothing, but it writes out a csv file that contains the results of running
    """

    available_zones = ['TopFloor_Plenum', 'MidFloor_Plenum', 'FirstFloor_Plenum',
                       'Core_top', 'Core_mid', 'Core_bottom',
                       'Perimeter_top_ZN_3', 'Perimeter_top_ZN_2', 'Perimeter_top_ZN_1', 'Perimeter_top_ZN_4',
                       'Perimeter_bot_ZN_3', 'Perimeter_bot_ZN_2', 'Perimeter_bot_ZN_1', 'Perimeter_bot_ZN_4',
                       'Perimeter_mid_ZN_3', 'Perimeter_mid_ZN_2', 'Perimeter_mid_ZN_1', 'Perimeter_mid_ZN_4']
    control_zones = available_zones[3:]
    airloops = {'Core_top': "PACU_VAV_top", 'Core_mid': "PACU_VAV_mid", 'Core_bottom': "PACU_VAV_bot",
                'Perimeter_top_ZN_3': "PACU_VAV_top", 'Perimeter_top_ZN_2': "PACU_VAV_top",
                'Perimeter_top_ZN_1': "PACU_VAV_top", 'Perimeter_top_ZN_4': "PACU_VAV_top",
                'Perimeter_bot_ZN_3': "PACU_VAV_bot", 'Perimeter_bot_ZN_2': "PACU_VAV_bot",
                'Perimeter_bot_ZN_1': "PACU_VAV_bot", 'Perimeter_bot_ZN_4': "PACU_VAV_bot",
                'Perimeter_mid_ZN_3': "PACU_VAV_mid", 'Perimeter_mid_ZN_2': "PACU_VAV_mid",
                'Perimeter_mid_ZN_1': "PACU_VAV_mid", 'Perimeter_mid_ZN_4': "PACU_VAV_mid"}
    capacity = {'Core_top': 53, 'Core_mid': 53, 'Core_bottom': 53,
                'Perimeter_top_ZN_3': 11, 'Perimeter_top_ZN_2': 7,
                'Perimeter_top_ZN_1': 11, 'Perimeter_top_ZN_4': 7,
                'Perimeter_bot_ZN_3': 11, 'Perimeter_bot_ZN_2': 7,
                'Perimeter_bot_ZN_1': 11, 'Perimeter_bot_ZN_4': 7,
                'Perimeter_mid_ZN_3': 11, 'Perimeter_mid_ZN_2': 7,
                'Perimeter_mid_ZN_1': 11, 'Perimeter_mid_ZN_4': 7}

    # Add state variables that we care about
    eplus_extra_states = dict()
    eplus_extra_states.update(
        {("Heating Coil Electric Energy", f"{zone} VAV Box Reheat Coil"): f"{zone} vav heating" for zone in
         control_zones})
    eplus_extra_states.update(
        {("System Node Temperature", "PACU_VAV_top Supply Equipment Inlet Node"): f"Top supply temperature"})
    eplus_extra_states.update(
        {("System Node Temperature", "PACU_VAV_mid Supply Equipment Inlet Node"): f"Mid supply temperature"})
    eplus_extra_states.update(
        {("System Node Temperature", "PACU_VAV_bot Supply Equipment Inlet Node"): f"Bot supply temperature"})

    for name in [
        "Zone Air Relative Humidity",
        "Zone Total Internal Total Heating Energy", "Zone Total Internal Total Heating Rate",
        "Zone Air System Sensible Heating Energy", "Zone Air System Sensible Heating Rate",
        "Zone Air System Sensible Cooling Energy", "Zone Air System Sensible Cooling Rate",
        "Zone Windows Total Heat Gain Energy", "Zone Windows Total Heat Loss Energy",
        "Zone Windows Total Transmitted Solar Radiation Energy", "Surface Window Transmitted Solar Radiation Energy",
        "Surface Window Net Heat Transfer Energy"
    ]:
        eplus_extra_states.update(
            {(name, f"{zone}"):
                 name.replace("Zone", zone).replace("Surface", zone + " Surface") for zone in control_zones})

    eplus_extra_states.update({("Air System Electric Energy", airloop):
                                   f"{airloop} energy" for airloop in set(airloops.values())})
    eplus_extra_states[('Site Outdoor Air Drybulb Temperature', 'Environment')] = "outdoor temperature"
    eplus_extra_states[('Site Direct Solar Radiation Rate per Area', 'Environment')] = "site solar radiation"
    eplus_extra_states[('Facility Total HVAC Electric Demand Power', 'Whole Building')] = "total hvac"
    eplus_extra_states[('Schedule Value', 'Hours_of_operation')] = "operating status"

    model = Model(idf_file_name="./eplus_files/OfficeMedium_Denver.idf",
                  weather_file="./eplus_files/USA_CO_Denver-Aurora-Buckley.AFB_.724695_TMY3.epw",
                  eplus_naming_dict=eplus_extra_states)
    sim_days = 30
    sim_step = 4
    model.set_runperiod(sim_days, start_month=6, start_day=1)
    model.set_timestep(sim_step)

    for people_object in model.get_configuration("People"):
        model.add_configuration("Schedule:Compact",
                                {"Name": f"{people_object['Zone_or_ZoneList_Name']} Customized Occupancy Schedule",
                                 "Schedule Type Limits Name": "Any Number",
                                 "Field 1": "Through: 12/31",
                                 "Field 2": "For: Weekdays",
                                 "Field 3": "Until: 9:00,0.0",
                                 "Field 4": "Until: 17:00,0.0",
                                 "Field 5": "Until: 24:00,0.0", })

        model.edit_configuration("People",
                                 {"Name": people_object["Name"]},
                                 {"Number of People Calculation Method": "People",
                                  "Number of People": 1,
                                  "Number of People Schedule Name":
                                      f"{people_object['Zone_or_ZoneList_Name']} Customized Occupancy Schedule"})
    for key, _ in eplus_extra_states.items():
        model.add_configuration("Output:Variable",
                                {"Key Value": key[1], "Variable Name": key[0], "Reporting Frequency": "Timestep"})

    with tqdm.tqdm(total=sim_step * 24 * sim_days * 5) as pbar:
        with open(logfile_name, "w") as outfile:
            for stp in range(220, 245, 5):
                model.edit_configuration("Schedule:Compact",
                                         {"Name": f"CLGSETP_SCH_YES_OPTIMUM"},
                                         {"Field 8": f"{stp / 10:.1f}",
                                          "Field 17": f"{stp / 10:.1f}",
                                          "Field 26": f"{stp / 10:.1f}"})

                if random_occupants:
                    occupants = [capacity[zone] // 2 for zone in control_zones]

                for j, v in enumerate(occupants):
                    model.edit_configuration("Schedule:Compact",
                                             {"Name": f"{control_zones[j]} Customized Occupancy Schedule"},
                                             {"Field 4": f"Until: 17:00,{v}.0"})

                state = model.reset()
                pbar.update(1)
                current = flatten_dict(state)
                total_hvac = state["total hvac"]

                writer = csv.DictWriter(outfile, fieldnames=current.keys())
                if stp == 220:
                    writer.writeheader()
                writer.writerow(current)

                while not model.is_terminate():
                    actions = list()
                    if random_occupants and state["timestep"] % (24 * sim_step) == 0 and state["timestep"] != 0:
                        for i, zone in enumerate(control_zones):
                            new_occupant = np.random.randint(max(0, occupants[i] - 2),
                                                             min(capacity[zone], occupants[i] + 2))
                            if new_occupant == occupants[i]:
                                new_occupant = np.random.randint(max(0, occupants[i] - 2),
                                                                 min(capacity[zone], occupants[i] + 2))
                            occupants[i] = new_occupant

                    if random_occupants and state["time"].hour == 8 and state["operating status"] != 0:
                        for i, zone in enumerate(control_zones):
                            actions.append({
                                "priority": 0,
                                "component_type": "Schedule:Compact",
                                "control_type": "Schedule Value",
                                "actuator_key": f"{zone} Customized Occupancy Schedule",
                                "value": occupants[i],
                                "start_time": state['timestep']
                            })
                    if random_occupants and state["time"].hour == 16 and state["operating status"] != 0:
                        for i, zone in enumerate(control_zones):
                            actions.append({
                                "priority": 0,
                                "component_type": "Schedule:Compact",
                                "control_type": "Schedule Value",
                                "actuator_key": f"{zone} Customized Occupancy Schedule",
                                "value": 0,
                                "start_time": state['timestep']
                            })
                    state = model.step(actions)
                    new_value = flatten_dict(state)
                    writer.writerow(new_value)
                    pbar.update(1)
                    total_hvac += state["total hvac"]
                print(total_hvac)


def run_random_occupancy(logfile_name: str) -> None:
    """
    The run_random_occupancy function is used to run the EnergyPlus model with random occupancy and temperature
    setpoints. The function takes in a logfile name as an argument and runs the simulation for 30 days,
    with 4 timesteps per hour. The function also adds state variables that we care about, such as heating coil
    electric energy, system node temperature etc., and outputs them to a csv file specified by the logfile_name
    argument. The function also edits some of the configurations in order to make it more diverse. The generated
    csv file is used as the historical building operations for training the zone energy estimator.

    :param logfile_name: str: Specify the name of the logfile that will be created
    :return: Nothing, but it writes out a csv file that contains the results
    """

    available_zones = ['TopFloor_Plenum', 'MidFloor_Plenum', 'FirstFloor_Plenum',
                       'Core_top', 'Core_mid', 'Core_bottom',
                       'Perimeter_top_ZN_3', 'Perimeter_top_ZN_2', 'Perimeter_top_ZN_1', 'Perimeter_top_ZN_4',
                       'Perimeter_bot_ZN_3', 'Perimeter_bot_ZN_2', 'Perimeter_bot_ZN_1', 'Perimeter_bot_ZN_4',
                       'Perimeter_mid_ZN_3', 'Perimeter_mid_ZN_2', 'Perimeter_mid_ZN_1', 'Perimeter_mid_ZN_4']
    control_zones = available_zones[3:]
    airloops = {'Core_top': "PACU_VAV_top", 'Core_mid': "PACU_VAV_mid", 'Core_bottom': "PACU_VAV_bot",
                'Perimeter_top_ZN_3': "PACU_VAV_top", 'Perimeter_top_ZN_2': "PACU_VAV_top",
                'Perimeter_top_ZN_1': "PACU_VAV_top", 'Perimeter_top_ZN_4': "PACU_VAV_top",
                'Perimeter_bot_ZN_3': "PACU_VAV_bot", 'Perimeter_bot_ZN_2': "PACU_VAV_bot",
                'Perimeter_bot_ZN_1': "PACU_VAV_bot", 'Perimeter_bot_ZN_4': "PACU_VAV_bot",
                'Perimeter_mid_ZN_3': "PACU_VAV_mid", 'Perimeter_mid_ZN_2': "PACU_VAV_mid",
                'Perimeter_mid_ZN_1': "PACU_VAV_mid", 'Perimeter_mid_ZN_4': "PACU_VAV_mid"}
    capacity = {'Core_top': 53, 'Core_mid': 53, 'Core_bottom': 53,
                'Perimeter_top_ZN_3': 11, 'Perimeter_top_ZN_2': 7,
                'Perimeter_top_ZN_1': 11, 'Perimeter_top_ZN_4': 7,
                'Perimeter_bot_ZN_3': 11, 'Perimeter_bot_ZN_2': 7,
                'Perimeter_bot_ZN_1': 11, 'Perimeter_bot_ZN_4': 7,
                'Perimeter_mid_ZN_3': 11, 'Perimeter_mid_ZN_2': 7,
                'Perimeter_mid_ZN_1': 11, 'Perimeter_mid_ZN_4': 7}

    # Add state variables that we care about
    eplus_extra_states = dict()
    eplus_extra_states.update(
        {("Heating Coil Electric Energy", f"{zone} VAV Box Reheat Coil"): f"{zone} vav heating" for zone in
         control_zones})
    eplus_extra_states.update(
        {("System Node Temperature", "PACU_VAV_top Supply Equipment Inlet Node"): f"Top supply temperature"})
    eplus_extra_states.update(
        {("System Node Temperature", "PACU_VAV_mid Supply Equipment Inlet Node"): f"Mid supply temperature"})
    eplus_extra_states.update(
        {("System Node Temperature", "PACU_VAV_bot Supply Equipment Inlet Node"): f"Bot supply temperature"})

    for name in [
        "Zone Air Relative Humidity",
        "Zone Thermostat Heating Setpoint Temperature", "Zone Thermostat Cooling Setpoint Temperature",
        "Zone Total Internal Total Heating Energy", "Zone Total Internal Total Heating Rate",
        "Zone Air System Sensible Heating Energy", "Zone Air System Sensible Heating Rate",
        "Zone Air System Sensible Cooling Energy", "Zone Air System Sensible Cooling Rate",
        "Zone Windows Total Heat Gain Energy", "Zone Windows Total Heat Loss Energy",
        "Zone Windows Total Transmitted Solar Radiation Energy", "Surface Window Transmitted Solar Radiation Energy",
        "Surface Window Net Heat Transfer Energy"
    ]:
        eplus_extra_states.update(
            {(name, f"{zone}"): name.replace("Zone", zone).replace("Surface", zone + " Surface") for zone in
             control_zones})

    eplus_extra_states.update(
        {("Air System Electric Energy", airloop): f"{airloop} energy" for airloop in set(airloops.values())})
    eplus_extra_states[('Site Outdoor Air Drybulb Temperature', 'Environment')] = "outdoor temperature"
    eplus_extra_states[('Site Direct Solar Radiation Rate per Area', 'Environment')] = "site solar radiation"
    eplus_extra_states[('Facility Total HVAC Electric Demand Power', 'Whole Building')] = "total hvac"
    eplus_extra_states[('Schedule Value', 'Hours_of_operation')] = "operating status"
    eplus_extra_states[('Site Day Type Index', 'Environment')] = "day type"
    valid_daytypes = (2, 3, 4, 5, 6, 9, 10)

    model = Model(idf_file_name="./eplus_files/OfficeMedium_Denver.idf",
                  weather_file="./eplus_files/USA_CO_Denver-Aurora-Buckley.AFB_.724695_TMY3.epw",
                  eplus_naming_dict=eplus_extra_states)
    sim_days = 30
    sim_step = 1
    model.set_runperiod(sim_days, start_month=7, start_day=1)
    model.set_timestep(sim_step)

    for people_object in model.get_configuration("People"):
        model.add_configuration("Schedule:Compact",
                                {"Name": f"{people_object['Zone_or_ZoneList_Name']} Customized Occupancy Schedule",
                                 "Schedule Type Limits Name": "Any Number",
                                 "Field 1": "Through: 12/31",
                                 "Field 2": "For: Weekdays",
                                 "Field 3": "Until: 9:00",
                                 "Field 4": "0.0",
                                 "Field 5": "Until: 17:00",
                                 "Field 6": "0.0",
                                 "Field 7": "Until: 24:00",
                                 "Field 8": "0.0",
                                 "Field 9": "For: AllOtherDays",
                                 "Field 10": "Until: 24:00",
                                 "Field 11": "0.0", })

        model.edit_configuration("People",
                                 {"Name": people_object["Name"]},
                                 {"Number of People Calculation Method": "People",
                                  "Number of People": 1,
                                  "Number of People Schedule Name":
                                      f"{people_object['Zone_or_ZoneList_Name']} Customized Occupancy Schedule"})

        model.add_configuration("Schedule:Compact",
                                {"Name": f"{people_object['Zone_or_ZoneList_Name']} Customized ClgSetp Schedule",
                                 "Schedule Type Limits Name": "Temperature",
                                 "Field 1": "Through: 12/31",
                                 "Field 2": "For: SummerDesignDay",
                                 "Field 3": "Until: 5:00",
                                 "Field 4": "26.7",
                                 "Field 5": "Until: 6:00",
                                 "Field 6": "25.7",
                                 "Field 7": "Until: 9:00",
                                 "Field 8": "24.0",
                                 "Field 9": "Until: 17:00",
                                 "Field 10": "24.0",
                                 "Field 11": "Until: 22:00",
                                 "Field 12": "24.0",
                                 "Field 13": "Until: 24:00",
                                 "Field 14": "26.7",
                                 "Field 15": "For: Weekdays",
                                 "Field 16": "Until: 5:00",
                                 "Field 17": "26.7",
                                 "Field 18": "Until: 6:00",
                                 "Field 19": "25.6",
                                 "Field 20": "Until: 9:00",
                                 "Field 21": "24.0",
                                 "Field 22": "Until: 17:00",
                                 "Field 23": "24.0",
                                 "Field 24": "Until: 22:00",
                                 "Field 25": "24.0",
                                 "Field 26": "Until: 24:00",
                                 "Field 27": "26.7",
                                 "Field 28": "For: Saturday Sunday Holidays WinterDesignDay AllOtherDays",
                                 "Field 29": "Until: 24:00",
                                 "Field 30": "26.7"})

        model.add_configuration("Schedule:Compact",
                                {"Name": f"{people_object['Zone_or_ZoneList_Name']} Customized HtgSetp Schedule",
                                 "Schedule Type Limits Name": "Temperature",
                                 "Field 1": "Through: 12/31",
                                 "Field 2": "For: WinterDesignDay",
                                 "Field 3": "Until: 5:00",
                                 "Field 4": "15.6",
                                 "Field 5": "Until: 6:00",
                                 "Field 6": "17.6",
                                 "Field 7": "Until: 9:00",
                                 "Field 8": "21.0",
                                 "Field 9": "Until: 17:00",
                                 "Field 10": "21.0",
                                 "Field 11": "Until: 22:00",
                                 "Field 12": "21.0",
                                 "Field 13": "Until: 24:00",
                                 "Field 14": "15.6",
                                 "Field 15": "For: Weekdays",
                                 "Field 16": "Until: 5:00",
                                 "Field 17": "15.6",
                                 "Field 18": "Until: 6:00",
                                 "Field 19": "17.8",
                                 "Field 20": "Until: 9:00",
                                 "Field 21": "21.0",
                                 "Field 22": "Until: 17:00",
                                 "Field 23": "21.0",
                                 "Field 24": "Until: 22:00",
                                 "Field 25": "21.0",
                                 "Field 26": "Until: 24:00",
                                 "Field 27": "15.6",
                                 "Field 28": "For: Saturday Sunday Holidays SummerDesignDay AllOtherDays",
                                 "Field 29": "Until: 24:00",
                                 "Field 30": "15.6"})

        model.edit_configuration("ThermostatSetpoint:DualSetpoint",
                                 {"Name": f"{people_object['Zone_or_ZoneList_Name']} DualSPSched"},
                                 {"Cooling Setpoint Temperature Schedule Name":
                                      f"{people_object['Zone_or_ZoneList_Name']} Customized ClgSetp Schedule",
                                  "Heating Setpoint Temperature Schedule Name":
                                      f"{people_object['Zone_or_ZoneList_Name']} Customized HtgSetp Schedule"})

    for key, _ in eplus_extra_states.items():
        model.add_configuration("Output:Variable",
                                {"Key Value": key[1], "Variable Name": key[0], "Reporting Frequency": "Timestep"})

    with tqdm.tqdm(total=sim_step * 24 * sim_days * 1) as pbar:
        with open(logfile_name, "w") as outfile:

            occupants = [capacity[zone] // 2 for zone in control_zones]

            for j, v in enumerate(occupants):
                model.edit_configuration("Schedule:Compact",
                                         {"Name": f"{control_zones[j]} Customized Occupancy Schedule"},
                                         {"Field 6": f"{v}.0"})

            state = model.reset()
            pbar.update(1)
            current = flatten_dict(state)
            total_hvac = state["total hvac"]

            writer = csv.DictWriter(outfile, fieldnames=current.keys())
            writer.writeheader()
            writer.writerow(current)
            temp_switcher = {5: [25.7, 17.8], 6: [24.0, 21.0], 9: [24.0, 23.0], 17: [24.0, 21.0], 22: [26.7, 15.6]}

            while not model.is_terminate():
                actions = list()

                if state["time"].hour == 8 and int(state["day type"]) in valid_daytypes:
                    for i, zone in enumerate(control_zones):
                        new_occupant = np.random.randint(max(1, occupants[i] - 2),
                                                         min(capacity[zone], occupants[i] + 2))
                        if new_occupant == occupants[i]:
                            new_occupant = np.random.randint(max(1, occupants[i] - 2),
                                                             min(capacity[zone], occupants[i] + 2))
                        occupants[i] = new_occupant

                        actions.append({
                            "priority": 0,
                            "component_type": "Schedule:Compact",
                            "control_type": "Schedule Value",
                            "actuator_key": f"{zone} Customized Occupancy Schedule",
                            "value": occupants[i],
                            "start_time": state['timestep']
                        })

                if state["time"].hour == 16:
                    for i, zone in enumerate(control_zones):
                        actions.append({
                            "priority": 0,
                            "component_type": "Schedule:Compact",
                            "control_type": "Schedule Value",
                            "actuator_key": f"{zone} Customized Occupancy Schedule",
                            "value": 0,
                            "start_time": state['timestep']
                        })

                if state["time"].hour + 1 in temp_switcher and int(state["day type"]) in valid_daytypes:
                    clg, htg = temp_switcher[state["time"].hour + 1]
                    clg = np.random.random() * 4 + 20
                    htg = clg - 1
                    for i, zone in enumerate(control_zones):
                        actions.append({
                            "priority": 0,
                            "component_type": "Schedule:Compact",
                            "control_type": "Schedule Value",
                            "actuator_key": f"{zone} Customized ClgSetp Schedule",
                            "value": clg if isinstance(clg, float) else clg[i],
                            "start_time": state['timestep']
                        })
                        actions.append({
                            "priority": 0,
                            "component_type": "Schedule:Compact",
                            "control_type": "Schedule Value",
                            "actuator_key": f"{zone} Customized HtgSetp Schedule",
                            "value": htg if isinstance(htg, float) else htg[i],
                            "start_time": state['timestep']
                        })

                state = model.step(actions)
                new_value = flatten_dict(state)
                writer.writerow(new_value)
                pbar.update(1)
                total_hvac += state["total hvac"]
            print(total_hvac)


if __name__ == '__main__':
    run_random_occupancy("simulated_log.csv")
