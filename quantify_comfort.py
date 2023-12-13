from cobs.model import Model
import numpy as np
import datetime
import utility

Model.set_energyplus_folder(utility.eplus_location)

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
temp_target = {'Core_top': 21.8, 'Core_mid': 22.2, 'Core_bottom': 22.0,
               'Perimeter_top_ZN_3': 21.7, 'Perimeter_top_ZN_2': 21.5,
               'Perimeter_top_ZN_1': 21.6, 'Perimeter_top_ZN_4': 21,
               'Perimeter_bot_ZN_3': 20.9, 'Perimeter_bot_ZN_2': 20.9,
               'Perimeter_bot_ZN_1': 21.4, 'Perimeter_bot_ZN_4': 21.3,
               'Perimeter_mid_ZN_3': 21.8, 'Perimeter_mid_ZN_2': 20.6,
               'Perimeter_mid_ZN_1': 21.5, 'Perimeter_mid_ZN_4': 21.2}

# Add state variables that we care about
eplus_extra_states = {("Zone Air Relative Humidity", zone): f"{zone} humidity" for zone in available_zones}
eplus_extra_states.update(
    {("Heating Coil Electric Energy", f"{zone} VAV Box Reheat Coil"): f"{zone} vav energy" for zone in
     available_zones})
eplus_extra_states.update(
    {("Air System Electric Energy", airloop): f"{airloop} energy" for airloop in set(airloops.values())})
eplus_extra_states[('Site Outdoor Air Drybulb Temperature', 'Environment')] = "outdoor temperature"
eplus_extra_states[('Site Direct Solar Radiation Rate per Area', 'Environment')] = "site solar radiation"
eplus_extra_states[('Facility Total HVAC Electric Demand Power', 'Whole Building')] = "total hvac"
eplus_extra_states[('Schedule Value', 'Hours_of_operation')] = "operating"


def quantify_comfort() -> None:
    """
    The quantify_comfort function calculates the PPD of a building model.

    :return: The total energy consumption and the average pmv and ppd
    """
    model = Model(idf_file_name="./eplus_files/OfficeMedium_Denver.idf",
                  weather_file="./eplus_files/USA_CO_Denver-Aurora-Buckley.AFB_.724695_TMY3.epw",
                  eplus_naming_dict=eplus_extra_states)
    model.set_runperiod(365)
    model.set_timestep(4)

    for people_object in model.get_configuration("People"):
        model.add_configuration("Schedule:Constant",
                                {"Name": f"{people_object['Zone_or_ZoneList_Name']} Customized Occupancy Schedule",
                                 "Schedule Type Limits Name": "Any Number",
                                 "Hourly Value": 0})

        model.edit_configuration("People",
                                 {"Name": people_object["Name"]},
                                 {"Number of People Calculation Method": "People",
                                  "Number of People": 1,
                                  "Number of People Schedule Name": f"{people_object['Zone_or_ZoneList_Name']} Customized Occupancy Schedule"})

        model.add_configuration("Schedule:Constant",
                                {"Name": f"{people_object['Zone_or_ZoneList_Name']} Customized Htg Schedule",
                                 "Schedule Type Limits Name": "Any Number",
                                 "Hourly Value": 21})
        model.add_configuration("Schedule:Constant",
                                {"Name": f"{people_object['Zone_or_ZoneList_Name']} Customized Clg Schedule",
                                 "Schedule Type Limits Name": "Any Number",
                                 "Hourly Value": 24})

        model.edit_configuration("ThermostatSetpoint:DualSetpoint",
                                 {"Name": f'{people_object["Name"]} DualSPSched'},
                                 {
                                     "Heating Setpoint Temperature Schedule Name": f"{people_object['Zone_or_ZoneList_Name']} Customized Htg Schedule",
                                     "Cooling Setpoint Temperature Schedule Name": f"{people_object['Zone_or_ZoneList_Name']} Customized Clg Schedule"})

    for key, _ in eplus_extra_states.items():
        model.add_configuration("Output:Variable",
                                {"Key Value": key[1], "Variable Name": key[0], "Reporting Frequency": "Timestep"})

    dist = np.ones(len(control_zones)) / len(control_zones)
    n_occupants = 100

    occupants = np.random.default_rng().multinomial(n_occupants, dist)

    state = model.reset()
    energy_consumption = state["total hvac"]
    counter = 0
    abs_pmv_sum = {k: 0 for k in state["PMV"].keys()}
    ppd_sum = {k: 0 for k in state["PMV"].keys()}
    pmv_vios = {k: 0 for k in state["PMV"].keys()}

    while not model.is_terminate():
        counter_incre = False
        for k in state["PMV"].keys():
            if state['occupancy'][k] != 0:
                abs_pmv_sum[k] += np.absolute(state["PMV"][k])
                ppd_sum[k] += state["PPD"][k]
                if np.absolute(state["PMV"][k]) >= 0.5:
                    pmv_vios[k] += 1
                counter_incre = True
        if counter_incre:
            counter += 1

        actions = list()

        next_time = state['time'] + datetime.timedelta(minutes=15)
        for i, zone in enumerate(control_zones):
            occupant = 0
            if float(state['operating']) != 0.0 and 9 <= next_time.hour < 17:
                occupant = occupants[i]

            actions.append({"priority": 0,
                            "component_type": "Schedule:Constant",
                            "control_type": "Schedule Value",
                            "actuator_key": f"{zone} Customized Occupancy Schedule",
                            "value": occupant,
                            "start_time": state['timestep']})

            actions.append({"priority": 0,
                            "component_type": "Schedule:Constant",
                            "control_type": "Schedule Value",
                            "actuator_key": f"{zone} Customized Htg Schedule",
                            "value": temp_target[zone] - 1,
                            "start_time": state['timestep']})
            actions.append({"priority": 0,
                            "component_type": "Schedule:Constant",
                            "control_type": "Schedule Value",
                            "actuator_key": f"{zone} Customized Clg Schedule",
                            "value": temp_target[zone] + 1,
                            "start_time": state['timestep']})

        state = model.step(actions)
        energy_consumption += state["total hvac"]

    print(energy_consumption)
    print(ppd_sum)


if __name__ == '__main__':
    quantify_comfort()
