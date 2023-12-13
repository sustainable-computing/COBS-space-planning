# from pyscipopt import Model, quicksum
from gurobipy import GRB, quicksum, MVar
import gurobipy as gp
from gurobi_ml import add_predictor_constr
import pandas as pd
import numpy as np
import pickle
import torch
from tqdm import tqdm
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from argparse import ArgumentParser
from ondemand_assignment import bestfit_energy, uniform_ratio, uniform_number, random
from icnn import ICNN
from utility import str_to_bool, to_numpy, to_torch

parser = ArgumentParser()
parser.add_argument("--estimator",
                    help="Define what estimator should be used for Q_zone estimation, possible choices: linear, rf, nn",
                    choices=["linear", "rf", "nn"], type=str, default="linear")
parser.add_argument("--warm_start", help="Define whether to use greedy approach for warm starting solver",
                    type=str_to_bool, default=True)
parser.add_argument("--num_long_term", help="Define how many long term occupants are in the building", type=int,
                    default=100)
parser.add_argument("--seed", help="Define random seed to use", type=int, default=999)
parser.add_argument("--thermal_threshold", help="Define hard constraint on thermal comfort score", type=float,
                    default=0.8)
parser.add_argument("--min_group_size", help="Define minimum number of occupants in the group", type=int, default=1)
parser.add_argument("--max_group_size", help="Define maximum number of occupants in the group", type=int, default=4)
parser.add_argument("--min_temp_set", help="Define minimum thermostat setpoint", type=float, default=20)
parser.add_argument("--max_temp_set", help="Define maximum thermostat setpoint", type=float, default=26)
parser.add_argument("--gap_tolerance", help="Define Dual and Primal bound gap tolerance", type=float, default=1e-4)
parser.add_argument("--special", help="Run assignment with special rule, choices: none, uniform_number, uniform_ratio, random",
                    choices=["none", "uniform_number", "uniform_ratio", "random"], type=str, default="none")
parser.add_argument("--with_zone_temp", help="Estimate Q with adjacent zone temperature or not", type=str_to_bool,
                    default=False)
parser.add_argument("--nn_shape", help="For NN estimator, give a python list style layer design", type=str,
                    default="[100]")
parser.add_argument("--save_result", help="Overwrite the result dump or not", type=str_to_bool, default=True)
args = parser.parse_args()


def logger(model, where):
    """
    The logger function is a callback function that will be called at each node of the branch-and-bound tree.
    It records the number of nodes explored, best bound, best solution found so far, and time elapsed.

    :param model: Access the model object
    :param where: Specify when the callback function is called
    :return: None
    """
    global vs
    if where == GRB.Callback.MIP:
        if model.cbGet(gp.GRB.Callback.MIP_NODCNT) != 0:
            vs.append([model.cbGet(gp.GRB.Callback.MIP_NODCNT),
                       model.cbGet(gp.GRB.Callback.MIP_OBJBND),
                       model.cbGet(gp.GRB.Callback.MIP_OBJBST),
                       model.cbGet(gp.GRB.Callback.RUNTIME),
                       model.cbGet(gp.GRB.Callback.WORK)])


def train_estimator_and_solve_minlp(log_file: str, base_energy="historical_base.pkl"):
    """
    The train_estimator_and_solve_minlp function trains an estimator on the data in log_file and then uses that
    estimator to solve a MINLP problem. The function returns the solution of the MINLP problem as well as some other
    information about how it was solved.

    :param log_file: str: Specify the path to the csv file containing all of the data
    :param base_energy: Specify the base energy consumption of each zone
    :return: None
    """
    global vs
    np.random.seed(args.seed)
    df = pd.read_csv(log_file, date_format=["time"])

    # Building information
    control_zones = ['Core_top', 'Core_mid', 'Core_bottom',
                     'Perimeter_top_ZN_3', 'Perimeter_top_ZN_2', 'Perimeter_top_ZN_1', 'Perimeter_top_ZN_4',
                     'Perimeter_bot_ZN_3', 'Perimeter_bot_ZN_2', 'Perimeter_bot_ZN_1', 'Perimeter_bot_ZN_4',
                     'Perimeter_mid_ZN_3', 'Perimeter_mid_ZN_2', 'Perimeter_mid_ZN_1', 'Perimeter_mid_ZN_4']
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

    out_temps = list()
    out_night_temps = list()

    # Preprocess training data
    values = list()
    for day in range(len(df) // 24):
        data = df[day * 24:day * 24 + 24]
        # Ignore holidays
        if int(data["time"].iloc[0][8:10]) in (16, 22, 23, 29, 30, 17, 24):
            continue

        # Capture all features
        for zone_name in control_zones:
            q_hvac = (data[f"{zone_name} Air System Sensible Cooling Rate"] -
                      data[f"{zone_name} Air System Sensible Heating Rate"]).mean()
            out_temp = data[5:20][f"outdoor temperature"].mean()
            out_night_temp = (data[0:5][f"outdoor temperature"].sum(
            ) + data[20:][f"outdoor temperature"].sum()) / 9
            if zone_name == control_zones[0] and len(out_temps) < 8:
                out_temps.append(out_temp)
                out_night_temps.append(out_night_temp)

            zone_temps = [data[f"{name} Thermostat Cooling Setpoint Temperature"].values[12] for name in control_zones]
            values.append([zone_name, data[f"temperature {zone_name}"].values[5],
                           data[f"{zone_name} Thermostat Cooling Setpoint Temperature"].values[12],
                           data["site solar radiation"].sum(), out_temp, data[f"occupancy {zone_name}"].max(), q_hvac,
                           out_night_temp] + zone_temps)

    # Preprocess training features
    values = pd.DataFrame(values,
                          columns=["Zone", "Pre temp", "Set temp", "Solar", "Out avg temp", "Occupants", "Total Energy",
                                   "Out night temp"] + control_zones)
    values["Temp diff"] = values["Pre temp"] - values["Set temp"]
    values["Out diff"] = values["Out avg temp"] - values["Pre temp"]
    values["Out set diff"] = values["Out avg temp"] - values["Set temp"]

    with open(base_energy, "rb") as infile:
        zone_hvac_energy_params, zone_vacancy_energy_bases = pickle.load(infile)
    avg_out_temp = np.mean(out_temps)
    avg_out_night_temp = np.mean(out_night_temps)

    linear_model = dict()
    linear_model_eval = dict()
    # Train estimators using various ML models
    for i, zone_name in tqdm(enumerate(control_zones), total=len(control_zones)):
        data = values[values["Zone"] == zone_name]
        data = data[data["Occupants"] != 0]
        if not args.with_zone_temp:
            x = data[["Out night temp", "Set temp", "Out avg temp", "Occupants"]].values
        else:
            x = data[["Out night temp", "Set temp", "Out avg temp", "Occupants"] + zone_adj[zone_name]].values
        y = data["Total Energy"].values
        if args.estimator == "linear":
            regressor = LinearRegression().fit(x, y)
            best_regressor = regressor
        else:
            counter = 0
            best_regressor = None
            while True:
                if args.estimator == "rf":
                    regressor = RandomForestRegressor().fit(x, y)
                elif args.estimator == "nn":
                    regressor = MLPRegressor(eval(args.nn_shape), solver="adam", max_iter=50000).fit(x, y)
                else:
                    raise NotImplementedError
                if best_regressor is None or regressor.score(x, y) > best_regressor.score(x, y):
                    best_regressor = regressor
                if regressor.score(x, y) >= 0.8 or counter == 10:
                    break
                counter += 1

        # Save the model for MIP
        linear_model[zone_name] = best_regressor
        linear_model_eval[zone_name] = (
            best_regressor.score(x, y), np.abs(best_regressor.predict(x) - y).mean(),
            np.power(best_regressor.predict(x) - y, 2).mean())

    # Generate long term occupants randomly
    group_sizes = list()
    while np.sum(group_sizes) <= args.num_long_term - args.max_group_size:
        group_sizes.append(np.random.randint(args.min_group_size, args.max_group_size + 1))
    group_sizes.append(args.num_long_term - np.sum(group_sizes))

    group_sizes = np.array(group_sizes)

    people_prefer_temp_mean = np.random.random(group_sizes.size) * (
            args.max_temp_set - args.min_temp_set) + args.min_temp_set

    thermal_threshold = np.sqrt(-2 * np.log(args.thermal_threshold))
    temp = thermal_threshold * 1.5
    thermal_lb = people_prefer_temp_mean - temp
    thermal_lb[thermal_lb < args.min_temp_set] = args.min_temp_set
    thermal_ub = people_prefer_temp_mean + temp
    thermal_ub[thermal_ub > args.max_temp_set] = args.max_temp_set

    # Setup the MIP problem and Gurobi solver
    m = gp.Model()
    m.setParam('MIPGap', args.gap_tolerance)
    m.setParam('IntFeasTol', 1e-9)
    objective = 0

    # Group assignment
    x = list()
    for i in range(len(control_zones)):
        row = list()
        for j in range(group_sizes.size):
            row.append(m.addVar(vtype='B', name=f"Assignment group {j} zone {i}"))
        x.append(row)

    avg_out_temp_var = m.addVar(lb=avg_out_temp, ub=avg_out_temp)
    avg_out_night_temp_var = m.addVar(lb=avg_out_night_temp, ub=avg_out_night_temp)

    t = list()
    for i in range(len(control_zones)):
        t.append(m.addVar(vtype='C', lb=args.min_temp_set, ub=args.max_temp_set, name=f"temp set zone {i}"))

    e = list()
    e_pred = list()
    for i in range(len(control_zones)):
        e.append(m.addVar(vtype='C', name=f"Zone {i} energy true estimation"))
        e_pred.append(m.addVar(vtype='C', name=f"Zone {i} energy occupied estimation"))

    num_occupants_in_zones = list()
    helper = list()
    for i, name in enumerate(control_zones):
        num_occupants_in_zones.append(m.addVar(vtype='I', name=f"Zone {i} total occupants"))
        helper.append(m.addVar(vtype='B', name=f"Zone {i} occupantion"))
        m.addConstr(num_occupants_in_zones[i] == quicksum(x[i][j] * group_sizes[j] for j in range(group_sizes.size)))
        m.addConstr(num_occupants_in_zones[i] <= capacity[name], name=f"Capacity limit zone {i}")
        m.addGenConstrOr(helper[i], [x[i][j] for j in range(group_sizes.size)])

    # Add constraints on matrix X
    for j in range(group_sizes.size):
        for i in range(len(control_zones)):
            m.addConstr(t[i] * x[i][j] <= thermal_ub[j] * x[i][j],
                        name=f"Zone {i} Thermal for Group {j} constraint (ub)")
            m.addConstr(-t[i] * x[i][j] <= -thermal_lb[j] * x[i][j],
                        name=f"Zone {i} Thermal for Group {j} constraint (lb)")
        m.addConstr(quicksum(x[i][j] for i in range(len(control_zones))) == 1,
                    name=f"Group {j} only assign to one zone constraint")
        # m.addSOS(GRB.SOS_TYPE1, [x[i][j] for i in range(len(control_zones))])

    # Estimate the energy consumption
    for i, name in enumerate(control_zones):
        var_list = [avg_out_night_temp_var, t[i], avg_out_temp_var, num_occupants_in_zones[i]]
        if args.with_zone_temp:
            for adj_name in zone_adj[name]:
                var_list.append(t[control_zones.index(adj_name)])
        input_vars = MVar.fromlist([var_list])
        output_vars = MVar.fromlist([[e_pred[i]]])
        add_predictor_constr(m, linear_model[name], input_vars, output_vars)
        # Set objective to minimize the estimated total zone energy consumption
        m.addConstr(e[i] * (1 - helper[i]) == zone_vacancy_energy_bases[i] * (1 - helper[i]))
        m.addConstr(e[i] * helper[i] == e_pred[i] * helper[i])

    m.setObjective(quicksum(e[i] for i in range(len(control_zones))), GRB.MINIMIZE)
    # m.addCons(objvar >= quicksum(e[i] for i in range(len(control_zones))))

    # Warm start the solver by using the result obtained when treating each long-term occupants as short-term occupants
    # and assign them using various online algorithms
    if args.warm_start or args.special != "none":
        avg_out_temp_var.start = avg_out_temp
        avg_out_night_temp_var.start = avg_out_night_temp
        # x, t, e, e_pred, num_occupants_in_zones, helper

        ws_x = list()
        for i in range(len(control_zones)):
            ws_x.append([0] * group_sizes.size)

        ws_occupants = [0] * len(control_zones)
        ws_temp = [args.max_temp_set] * len(control_zones)
        ws_occupied_thermal_lb = [args.min_temp_set] * len(control_zones)
        ws_occupied_thermal_ub = [args.max_temp_set] * len(control_zones)
        zone_thermals = [list() for _ in range(len(control_zones))]

        for j in range(group_sizes.size):
            if args.special != "none":
                ws_f = eval(args.special)
            else:
                ws_f = bestfit_energy
            ws_result = ws_f(control_zones, capacity, zone_adj, args.with_zone_temp, group_sizes[j], thermal_lb[j],
                             thermal_ub[j],
                             linear_model, zone_vacancy_energy_bases, avg_out_temp, avg_out_night_temp,
                             ws_occupants, ws_temp, ws_occupied_thermal_lb, ws_occupied_thermal_ub,
                             zone_thermals=zone_thermals)
            if ws_result is None:
                raise ValueError
            i, ws_temp_sol = ws_result
            if args.special != "none":
                zone_thermals[i].append(people_prefer_temp_mean[j] * group_sizes[j])
            ws_occupied_thermal_lb[i] = max(ws_occupied_thermal_lb[i], thermal_lb[j])
            ws_occupied_thermal_ub[i] = min(ws_occupied_thermal_ub[i], thermal_ub[j])
            ws_occupants[i] += group_sizes[j]
            ws_temp[i] = ws_temp_sol
            ws_x[i][j] = 1

        # Assign variables with online version problem's solution
        for i in range(len(control_zones)):
            for j in range(group_sizes.size):
                x[i][j].start = ws_x[i][j]

            t[i].start = ws_temp[i]
            ws_v = [avg_out_night_temp, ws_temp[i], avg_out_temp, ws_occupants[i]]
            if args.with_zone_temp:
                for adj_name in zone_adj[control_zones[i]]:
                    ws_v.append(ws_temp[control_zones.index(adj_name)])
            occupied_energy = linear_model[control_zones[i]].predict([ws_v])[0]
            e[i].start = occupied_energy if ws_occupants[i] != 0 else zone_vacancy_energy_bases[i]
            e_pred[i].start = occupied_energy
            num_occupants_in_zones[i].start = ws_occupants[i]
            helper[i].start = 0 if ws_occupants[i] == 0 else 1

        m.update()

    # Solve the MIP and reformat the solution to save locally
    if args.special == "none":
        vs = list()
        m.optimize(logger)

        solution_x = np.zeros((len(control_zones), group_sizes.size))
        for i in range(len(control_zones)):
            for j in range(group_sizes.size):
                solution_x[i, j] = int(np.round(x[i][j].x))
        occupants = np.zeros(len(control_zones))
        for i in range(len(control_zones)):
            occupants[i] = int(np.round(num_occupants_in_zones[i].x))
        solution_t = np.zeros(len(control_zones))
        for i in range(len(control_zones)):
            solution_t[i] = t[i].x
    else:
        solution_x = np.zeros((len(control_zones), group_sizes.size))
        for i in range(len(control_zones)):
            for j in range(group_sizes.size):
                solution_x[i, j] = int(np.round(ws_x[i][j]))
        occupants = np.zeros(len(control_zones))
        for i in range(len(control_zones)):
            occupants[i] = int(np.round(ws_occupants[i]))
        solution_t = np.zeros(len(control_zones))
        for i in range(len(control_zones)):
            solution_t[i] = ws_temp[i]

    for i in range(len(control_zones)):
        print(f"Zone {control_zones[i]}: {occupants[i]}")

    # Save results for long-term occupants
    outname = (f"results/gurobi_seed_{args.seed}_offline_{args.num_long_term}_estimator"
               f"_{args.estimator}_adjacentzone_{str(args.with_zone_temp)}.pkl")
    if args.special != "none":
        outname = outname.replace(f"estimator_{args.estimator}", f"special_{args.special}")

    if args.save_result:
        with open(outname, "wb") as outfile:
            pickle.dump({"linear_models": linear_model,
                         "linear_model_evals": linear_model_eval,
                         "vacant_base": zone_vacancy_energy_bases,
                         "group_sizes": group_sizes,
                         "people_prefer_temp_mean": people_prefer_temp_mean,
                         "avg_out_temp": avg_out_temp,
                         "avg_out_night_temp": avg_out_night_temp,
                         "thermal_lb": thermal_lb,
                         "thermal_ub": thermal_ub,
                         "solution_x": solution_x,
                         "solution_t": solution_t,
                         "occupants": occupants,
                         "objective": m.ObjVal if args.special == "none" else 0,
                         "adj_zones": args.with_zone_temp}, outfile)

    with open(outname.replace("results", "results/solver_status"), "wb") as outfile:
        pickle.dump(vs, outfile)

if __name__ == "__main__":
    vs = list()
    os.makedirs("results/solver_status", exist_ok=True)
    train_estimator_and_solve_minlp("simulated_log.csv")