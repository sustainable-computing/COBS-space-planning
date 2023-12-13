import numpy as np
from cobs.model import Model
from datetime import datetime, timedelta
import utility
import pickle
import tqdm
import os
import sys
import glob
from argparse import ArgumentParser
from ondemand_assignment import *
from utility import str_to_bool

sys.path.insert(0, utility.file_path)
Model.set_energyplus_folder(utility.eplus_location)


def performance_eval(result_file="evaluate_result.csv"):
    """
    The performance_eval function is used to evaluate the performance of a given long-term assignment and selected
    short-term assignment based on the command-line arguments. The command-line arguments specs located at the bottom
    of this code file. It takes in the name of a pickle file containing an offline solution, and runs it through
    EnergyPlus for a specified number of days. The function then calculates various metrics such as energy consumption,
    thermal comfort, and utilization rate. It also outputs occupancy changes over time for each zone.

    :param result_file: Specify the file name of the long-term optimization result
    :return: None
    """
    # Setup the building
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
    zone_adj = {'Core_top': ['Core_mid', 'Perimeter_top_ZN_1', 'Perimeter_top_ZN_2', 'Perimeter_top_ZN_3',
                             'Perimeter_top_ZN_4'],
                'Core_mid': ['Core_top', 'Core_bottom', 'Perimeter_mid_ZN_1', 'Perimeter_mid_ZN_2',
                             'Perimeter_mid_ZN_3', 'Perimeter_mid_ZN_4'],
                'Core_bottom': ['Core_mid', 'Perimeter_bot_ZN_1', 'Perimeter_bot_ZN_2', 'Perimeter_bot_ZN_3',
                                'Perimeter_bot_ZN_4'],
                'Perimeter_top_ZN_1': ['Core_top', 'Perimeter_mid_ZN_1', 'Perimeter_top_ZN_2', 'Perimeter_top_ZN_4'],
                'Perimeter_top_ZN_2': ['Core_top', 'Perimeter_mid_ZN_2', 'Perimeter_top_ZN_1', 'Perimeter_top_ZN_3'],
                'Perimeter_top_ZN_3': ['Core_top', 'Perimeter_mid_ZN_3', 'Perimeter_top_ZN_2', 'Perimeter_top_ZN_4'],
                'Perimeter_top_ZN_4': ['Core_top', 'Perimeter_mid_ZN_4', 'Perimeter_top_ZN_1', 'Perimeter_top_ZN_3'],
                'Perimeter_mid_ZN_1': ['Core_mid', 'Perimeter_top_ZN_1', 'Perimeter_bot_ZN_1', 'Perimeter_mid_ZN_2',
                                       'Perimeter_mid_ZN_4'],
                'Perimeter_mid_ZN_2': ['Core_mid', 'Perimeter_top_ZN_2', 'Perimeter_bot_ZN_2', 'Perimeter_mid_ZN_1',
                                       'Perimeter_mid_ZN_3'],
                'Perimeter_mid_ZN_3': ['Core_mid', 'Perimeter_top_ZN_3', 'Perimeter_bot_ZN_3', 'Perimeter_mid_ZN_2',
                                       'Perimeter_mid_ZN_4'],
                'Perimeter_mid_ZN_4': ['Core_mid', 'Perimeter_top_ZN_4', 'Perimeter_bot_ZN_4', 'Perimeter_mid_ZN_1',
                                       'Perimeter_mid_ZN_3'],
                'Perimeter_bot_ZN_1': ['Core_bottom', 'Perimeter_mid_ZN_1', 'Perimeter_bot_ZN_2', 'Perimeter_bot_ZN_4'],
                'Perimeter_bot_ZN_2': ['Core_bottom', 'Perimeter_mid_ZN_2', 'Perimeter_bot_ZN_1', 'Perimeter_bot_ZN_3'],
                'Perimeter_bot_ZN_3': ['Core_bottom', 'Perimeter_mid_ZN_3', 'Perimeter_bot_ZN_2', 'Perimeter_bot_ZN_4'],
                'Perimeter_bot_ZN_4': ['Core_bottom', 'Perimeter_mid_ZN_4', 'Perimeter_bot_ZN_1', 'Perimeter_bot_ZN_3']}

    # Where to save the result
    processed = set()
    # Check result existence
    if not args.reprocess and os.path.isfile(result_file):
        with open(result_file, "r") as results:
            for line in results:
                if not line:
                    continue
                line = line.split(',')
                if line[1][1:-1] == args.online and line[2][1:-1] == str(args.num_visitor) and line[3][1:-1] == str(
                        args.days) and line[4][1:-1] == str(args.seed):
                    processed.add(line[0][1:-1])

    # Load models and long-term assignment
    pkls = sorted(glob.glob("results/gurobi*.pkl"))
    if args.designate_base:
        pkls = sorted(glob.glob(f"results/*{args.designate_base}*"))
    i = 0
    while i != len(pkls):
        if os.path.basename(pkls[i]) in processed or (not args.debug and "seed_999" in pkls[i]):
            pkls.pop(i)
        else:
            i += 1

    # Setup for multi-threading
    start = int(np.round(len(pkls) * args.n / args.parallel))
    end = int(np.round(len(pkls) * (args.n + 1) / args.parallel))
    pkls = pkls[start:end]
    os.makedirs(f"tmp/{args.job_name}_{args.online}_{args.parallel}-{args.n}", exist_ok=True)

    # Add state variables that we care about
    eplus_extra_states = dict()

    for name in [
        "Zone Air Relative Humidity",
        "Zone Total Internal Total Heating Energy", "Zone Total Internal Total Heating Rate",
        "Zone Air System Sensible Heating Energy", "Zone Air System Sensible Heating Rate",
        "Zone Air System Sensible Cooling Energy", "Zone Air System Sensible Cooling Rate",
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

    model = Model(idf_file_name="./eplus_files/OfficeMedium_Denver.idf",
                  weather_file="./eplus_files/USA_CO_Denver-Aurora-Buckley.AFB_.724695_TMY3.epw",
                  eplus_naming_dict=eplus_extra_states,
                  tmp_idf_path=f"tmp/{args.job_name}_{args.online}_{args.parallel}-{args.n}")
    sim_days = args.days
    if args.curves:
        sim_days = 1
    sim_step = 4
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
                                  "Number of People Schedule Name": f"{people_object['Zone_or_ZoneList_Name']} Customized Occupancy Schedule"})

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

    if args.curves:
        curves = list()

    with tqdm.tqdm(total=sim_step * 24 * sim_days * len(pkls)) as pbar:
        with open(result_file, "a") as outfile:
            for result_pkl in pkls:

                # Load long-term assignment
                with open(result_pkl, "rb") as infile:
                    offline_result = pickle.load(infile)

                for j, v in enumerate(offline_result["occupants"]):
                    model.edit_configuration("Schedule:Compact",
                                             {"Name": f"{control_zones[j]} Customized Occupancy Schedule"},
                                             {"Field 6": f"{int(v)}.0"})
                temp_switcher = {5: [25.7, 17.8], 6: [24.0, 21.0],
                                 9: [offline_result["solution_t"], offline_result["solution_t"] - 1], 17: [24.0, 21.0],
                                 22: [26.7, 15.6]}

                occupied_thermal_lb = np.array([args.min_temp_set] * len(control_zones), dtype=float)
                occupied_thermal_ub = np.array([args.max_temp_set] * len(control_zones), dtype=float)
                for i in range(len(control_zones)):
                    if offline_result["solution_x"][i, :].sum() == 0:
                        continue
                    occupied_thermal_lb[i] = offline_result["thermal_lb"][offline_result["solution_x"][i, :] == 1].max()
                    occupied_thermal_ub[i] = offline_result["thermal_ub"][offline_result["solution_x"][i, :] == 1].min()
                thermal_score = (np.exp(-np.power(((offline_result["solution_t"].dot(offline_result["solution_x"]) -
                                                    offline_result["people_prefer_temp_mean"]) / 1.5), 2) / 2) *
                                 offline_result["group_sizes"]).sum() / offline_result["group_sizes"].sum()
                thermal_score_min = np.exp(-np.power(((offline_result["solution_t"].dot(offline_result["solution_x"]) -
                                                       offline_result["people_prefer_temp_mean"]) / 1.5), 2) / 2).min()

                # Prepare for short-term occupants
                if args.energy:
                    state = model.reset()
                else:
                    state = {"total hvac": 0,
                             "timestep": 0,
                             "time": datetime(2000, 7, 1, 0, 0),
                             "operating status": 1}
                pbar.update(1)
                total_hvac = state["total hvac"]
                utilization = list()
                new_groups = list()
                reject_groups = list()
                reject_visitors = list()
                thermal = list()
                occupant_changes = list()
                thermal_min = list()

                while (args.energy and not model.is_terminate()) or (
                        not args.energy and state["timestep"] != model.get_total_timestep()):
                    actions = list()

                    # Start to take short-term group assignment requests
                    if state["time"].minute == 45 and state["time"].hour == 8 and state["operating status"]:
                        occupants = offline_result["occupants"].copy()
                        new_groups.append(0)
                        reject_groups.append(0)
                        reject_visitors.append(0)

                        if args.online != "none":
                            # Retrieve long-term occupants
                            temps = offline_result["solution_t"].copy()
                            occupied_thermal_lbs = occupied_thermal_lb.copy()
                            occupied_thermal_ubs = occupied_thermal_ub.copy()
                            possible_temps = None

                            # Simulate visitors
                            group_sizes = list()
                            while np.sum(group_sizes) <= args.num_visitor - args.max_group_size:
                                group_sizes.append(np.random.randint(args.min_group_size, args.max_group_size + 1))
                            if np.sum(group_sizes) != args.num_visitor:
                                group_sizes.append(args.num_visitor - np.sum(group_sizes))
                            group_sizes = np.array(group_sizes)
                            new_groups[-1] += group_sizes.size
                            people_prefer_temp_mean = np.random.random(group_sizes.size) * (
                                    args.max_temp_set - args.min_temp_set) + args.min_temp_set
                            thermal_threshold = np.sqrt(-2 * np.log(args.thermal_threshold))
                            temp = thermal_threshold * 1.5
                            thermal_lb = people_prefer_temp_mean - temp
                            thermal_lb[thermal_lb < args.min_temp_set] = args.min_temp_set
                            thermal_ub = people_prefer_temp_mean + temp
                            thermal_ub[thermal_ub > args.max_temp_set] = args.max_temp_set
                            zone_thermals = [list() for _ in range(len(control_zones))]

                            xs = np.concatenate([offline_result["solution_x"],
                                                 np.zeros((len(control_zones), people_prefer_temp_mean.size))], axis=1)
                            selected = [True] * offline_result["group_sizes"].size

                            # Choose algorithm
                            if args.online != "none":
                                online_function = eval(args.online)
                            if args.online == "online_minlp":
                                online_function = bestfit_energy

                            if args.curves:
                                thermal_curve = [thermal_score]
                                occupant_curve = [occupants.sum()]
                                rejection_curve = [0]
                                rejection_group_curve = [0]
                                rejection_rate = [0]

                            # Assign them to zones
                            for j in range(group_sizes.size):
                                ws_result = online_function(control_zones, capacity, zone_adj,
                                                            offline_result["adj_zones"], group_sizes[j], thermal_lb[j],
                                                            thermal_ub[j],
                                                            offline_result["linear_models"],
                                                            offline_result["vacant_base"],
                                                            offline_result["avg_out_temp"],
                                                            offline_result["avg_out_night_temp"],
                                                            occupants, temps, occupied_thermal_lbs,
                                                            occupied_thermal_ubs, possible_temps=possible_temps,
                                                            zone_thermals=zone_thermals)
                                if args.online == "online_minlp":
                                    ws_result = online_minlp(control_zones, capacity, zone_adj,
                                                             offline_result["adj_zones"], group_sizes[j],
                                                             thermal_lb[j], thermal_ub[j],
                                                             offline_result["linear_models"],
                                                             offline_result["vacant_base"],
                                                             offline_result["avg_out_temp"],
                                                             offline_result["avg_out_night_temp"],
                                                             occupants, temps, occupied_thermal_lbs,
                                                             occupied_thermal_ubs, possible_temps=possible_temps,
                                                             warm_start=ws_result)
                                if ws_result is None:
                                    reject_groups[-1] += 1
                                    reject_visitors[-1] += group_sizes[j]
                                    selected.append(False)

                                    if args.curves:
                                        thermal_curve.append(thermal_curve[-1])
                                        occupant_curve.append(occupant_curve[-1])
                                        rejection_curve.append(rejection_curve[-1] + group_sizes[j])
                                        rejection_group_curve.append(rejection_group_curve[-1] + 1)
                                        rejection_rate.append(rejection_group_curve[-1] / (j + 1) * 100)
                                else:
                                    if args.online != "bestfit_space":
                                        i, ws_temp_sol = ws_result
                                    elif args.online == "bestfit_space":
                                        i, ws_temp_sol, possible_temps = ws_result

                                    if args.online in ("uniform_number", "uniform_ratio"):
                                        zone_thermals[i].append(people_prefer_temp_mean[j] * group_sizes[j])
                                    occupied_thermal_lbs[i] = max(occupied_thermal_lbs[i], thermal_lb[j])
                                    occupied_thermal_ubs[i] = min(occupied_thermal_ubs[i], thermal_ub[j])
                                    occupants[i] += group_sizes[j]
                                    temps[i] = ws_temp_sol
                                    xs[i, offline_result["solution_x"].shape[1] + j] = 1
                                    selected.append(True)

                                    if args.curves:
                                        thermals = np.exp(-np.power(((temps.dot(
                                            xs[:, :offline_result["solution_x"].shape[1] + j + 1]) - np.concatenate(
                                            [offline_result["people_prefer_temp_mean"],
                                             people_prefer_temp_mean[:j + 1]])) / 1.5), 2) / 2)
                                        thermal_curve.append((thermals * np.concatenate(
                                            [offline_result["group_sizes"], group_sizes[:j + 1]]))[
                                                                 selected].sum() / occupants.sum())
                                        occupant_curve.append(occupant_curve[-1] + group_sizes[j])
                                        rejection_curve.append(rejection_curve[-1])
                                        rejection_group_curve.append(rejection_group_curve[-1])
                                        rejection_rate.append(rejection_rate[-1])

                                if args.curves:
                                    for _ in range(group_sizes[j] - 1):
                                        thermal_curve.append(thermal_curve[-1])
                                        occupant_curve.append(occupant_curve[-1])
                                        rejection_curve.append(rejection_curve[-1])
                                        rejection_group_curve.append(rejection_group_curve[-1])
                                        rejection_rate.append(rejection_rate[-1])

                            if args.curves:
                                curves.append([thermal_curve, occupant_curve, rejection_curve, rejection_group_curve,
                                               rejection_rate])

                            thermals = np.exp(-np.power(((temps.dot(xs) - np.concatenate(
                                [offline_result["people_prefer_temp_mean"], people_prefer_temp_mean])) / 1.5), 2) / 2)
                            thermal.append((thermals * np.concatenate([offline_result["group_sizes"], group_sizes]))[
                                               selected].sum() / occupants.sum())
                            thermal_min.append(thermals[selected].min())
                            temp_switcher[9] = [temps, temps - 1]
                        else:
                            thermal.append(thermal_score)
                            thermal_min.append(thermal_score_min)
                        utilization.append(occupants.sum() / 267)
                        occupant_changes.append(occupants.tolist())

                        # Update the simulator with the newest changes
                        for i, zone in enumerate(control_zones):
                            actions.append({
                                "priority": 0,
                                "component_type": "Schedule:Compact",
                                "control_type": "Schedule Value",
                                "actuator_key": f"{zone} Customized Occupancy Schedule",
                                "value": occupants[i],
                                "start_time": state['timestep']
                            })
                    if state["time"].minute == 45 and state["time"].hour == 16:
                        for i, zone in enumerate(control_zones):
                            actions.append({
                                "priority": 0,
                                "component_type": "Schedule:Compact",
                                "control_type": "Schedule Value",
                                "actuator_key": f"{zone} Customized Occupancy Schedule",
                                "value": 0,
                                "start_time": state['timestep']
                            })

                    if state["time"].minute == 45 and state["time"].hour + 1 in temp_switcher and state[
                        "time"].day % 7 not in (0, 1) and state["time"].day != 4:
                        clg, htg = temp_switcher[state["time"].hour + 1]
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

                    if args.energy:
                        state = model.step(actions)
                    else:
                        state["timestep"] += 1
                        state["time"] += timedelta(minutes=15)

                    pbar.update(1)
                    total_hvac += state["total hvac"]

                if args.curves:
                    continue
                out_line = (f'"{os.path.basename(result_pkl)}","{args.online}","{args.num_visitor}","{args.days}",'
                            f'"{args.seed}","{total_hvac}","{utilization}","{new_groups}","{reject_groups}",'
                            f'"{reject_visitors}","{thermal}","{thermal_min}","{offline_result["adj_zones"]}",'
                            f'"{offline_result["linear_model_evals"]}"')
                for v in occupant_changes:
                    out_line += f',"{v}"'
                out_line += '\n'
                outfile.write(out_line)

    if args.curves:
        with open(f"results/curves/{args.online}_seed_{args.seed}_{os.path.basename(result_pkl)}", "wb") as pklfile:
            pickle.dump(curves, pklfile)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--online",
                        help="Define what online algorithm should be used.",
                        choices=["bestfit_energy", "bestfit_space", "online_minlp", "uniform_number", "uniform_ratio", "random", "none"], type=str,
                        default="none")
    parser.add_argument("--num_visitor", help="Define in average how many visitors are in the building per day",
                        type=int, default=100)
    parser.add_argument("--seed", help="Define random seed to use", type=int, default=999)
    parser.add_argument("--days", help="Define days to simulate", type=int, default=14)
    parser.add_argument("--energy", help="Define whether simulate the energy consumption", type=str_to_bool,
                        default=True)
    parser.add_argument("--reprocess", help="Define whether redo simulation for finished jobs", type=str_to_bool,
                        default=True)
    parser.add_argument("--parallel", help="Define run on how many threads", type=int, default=1)
    parser.add_argument("--n", help="Define current thread id", type=int, default=0)
    parser.add_argument("--job_name", help="Define a name", type=str, default="experiment")
    parser.add_argument("--min_group_size", help="Define minimum number of occupants in the group", type=int, default=1)
    parser.add_argument("--max_group_size", help="Define maximum number of occupants in the group", type=int, default=4)
    parser.add_argument("--min_temp_set", help="Define minimum thermostat setpoint", type=float, default=20)
    parser.add_argument("--max_temp_set", help="Define maximum thermostat setpoint", type=float, default=26)
    parser.add_argument("--thermal_threshold", help="Define hard constraint on thermal comfort score", type=float,
                        default=0.8)
    parser.add_argument("--designate_base", help="Define offline MINLP solution file", type=str, default="")
    parser.add_argument("--curves", help="Log the evaluation changing trend", type=str_to_bool, default=False)
    parser.add_argument("--debug", help="Use specific node to debug", type=str_to_bool, default=False)
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs("results/curves", exist_ok=True)
    performance_eval()
