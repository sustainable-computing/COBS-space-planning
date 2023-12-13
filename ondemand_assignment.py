import numpy as np
from gurobipy import GRB, quicksum, MVar
import gurobipy as gp
from gurobi_ml import add_predictor_constr


def bestfit_energy(control_zones, capacity, zone_adj, use_adj, visitor_size, thermal_lb, thermal_ub,
                   occupied_models, empty_models, out_day_temp, out_night_temp,
                   occupants, temp, occupied_thermal_lb, occupied_thermal_ub, **kwargs):
    """
    The bestfit_energy function is used to assign a visitor group to a zone and set the temperature of that zone.
    The algorithm is to select the zone that has the minimum energy increase after accommodate the given group of
    occupants (needs to satisfy the zone capacity limit as well). This is the "BestFit-Energy" algorithm in paper.

    :param control_zones: Specify the zones that can be controlled by the hvac system
    :param capacity: Determine the maximum number of occupants in a zone
    :param zone_adj: Specify the adjacent zones of each zone
    :param use_adj: Indicate whether or not to use adjacent zone temperature as input for the model
    :param visitor_size: Determine the number of short term occupants in a group
    :param thermal_lb: Set the lower bound of temperature based on the short term occupants' preference
    :param thermal_ub: Set the upper bound of temperature based on the short term occupants' preference
    :param occupied_models: Predict the energy consumption of a zone given its temperature setpoint, occupancy, and more
    :param empty_models: Estimate the energy consumption of an empty zone
    :param out_day_temp: The average outdoor temperature during day time
    :param out_night_temp: The average outdoor temperature at night
    :param occupants: The number of occupants in each zone
    :param temp: Pass the current temperature of each zone
    :param occupied_thermal_lb: The lower bound of temperature setpoint for occupied zones
    :param occupied_thermal_ub: The upper bound of temperature setpoint for occupied zones
    :param **kwargs: Pass additional arguments to the function
    :return: The best zone to assign the visitor group, and the temperature setpoint, None if reject the occupant
    """
    candidate = list()
    for i, zone in enumerate(control_zones):
        # Skip zones that does not have enough capacity
        if visitor_size + occupants[i] <= capacity[zone]:
            # If the zone is occupied, we tend to use existing temperature
            if occupants[i] != 0:
                t = temp[i]
                if thermal_lb > temp[i] or temp[i] > thermal_ub:
                    lb = max(occupied_thermal_lb[i], thermal_lb)
                    ub = min(occupied_thermal_ub[i], thermal_ub)
                    if lb > ub:
                        continue
                    ts = np.linspace(lb, ub, 20)
                    stack_vals = [[out_night_temp] * ts.size, ts, [out_day_temp] * ts.size,
                                  [visitor_size + occupants[i]] * ts.size]
                    if use_adj:
                        for adj_name in zone_adj[zone]:
                            stack_vals.append([temp[control_zones.index(adj_name)]] * ts.size)
                    x = np.vstack(stack_vals).T
                    y = occupied_models[zone].predict(x)
                    t = ts[np.argmin(y)]

                v = [out_night_temp, t, out_day_temp, visitor_size + occupants[i]]
                if use_adj:
                    for adj_name in zone_adj[zone]:
                        v.append(temp[control_zones.index(adj_name)])
                # Estimate the energy increase due to having new group of visitors
                e = occupied_models[zone].predict([v])[0]
                v[3] -= visitor_size
                e -= occupied_models[zone].predict([v])[0]
            # If the zone is empty, assign the temperature to the value that satisfy visitor's thermal preference and minimize the energy consumption
            else:
                ts = np.linspace(thermal_lb, thermal_ub, 20)
                stack_vals = [[out_night_temp] * ts.size, ts, [out_day_temp] * ts.size, [visitor_size] * ts.size]
                if use_adj:
                    for adj_name in zone_adj[zone]:
                        stack_vals.append([temp[control_zones.index(adj_name)]] * ts.size)
                x = np.vstack(stack_vals).T
                y = occupied_models[zone].predict(x)
                # Use the best temperature setpoint
                t = ts[np.argmin(y)]
                # Estimate the energy increase
                e = np.min(y) - empty_models[i]
            candidate.append([int(occupants[i] == 0), e, i, t])

    # If no zone has the space, then the visitor request is rejected
    if not candidate:
        return None
    else:
        # Sort the available zones based on potential energy consumption increase in ascending order
        candidate.sort()
        return candidate[0][2], candidate[0][3]


def bestfit_space(control_zones, capacity, zone_adj, use_adj, visitor_size, thermal_lb, thermal_ub,
                  occupied_models, empty_models, out_day_temp, out_night_temp,
                  occupants, temp, occupied_thermal_lb, occupied_thermal_ub, possible_temps=None, **kwargs):
    # Check current temperature coverage
    """
    The bestfit_space function is used to assign a visitor group to a zone and set the temperature of that zone.
    The algorithm is to select the zone that has the maximum space available to accommodate the given group of
    occupants (needs to satisfy the zone capacity and thermal comfort limit as well).
    This is the "BestFit-Space" algorithm in paper.

    :param control_zones: Specify the zones that can be controlled by the hvac system
    :param capacity: Determine the maximum number of occupants in a zone
    :param zone_adj: Specify the adjacent zones of each zone
    :param use_adj: Indicate whether or not to use adjacent zone temperature as input for the model
    :param visitor_size: Determine the number of short term occupants in a group
    :param thermal_lb: Set the lower bound of temperature based on the short term occupants' preference
    :param thermal_ub: Set the upper bound of temperature based on the short term occupants' preference
    :param occupied_models: Predict the energy consumption of a zone given its temperature setpoint, occupancy, and more
    :param empty_models: Estimate the energy consumption of an empty zone
    :param out_day_temp: The average outdoor temperature during day time
    :param out_night_temp: The average outdoor temperature at night
    :param occupants: The number of occupants in each zone
    :param temp: Pass the current temperature of each zone
    :param occupied_thermal_lb: The lower bound of temperature setpoint for occupied zones
    :param occupied_thermal_ub: The upper bound of temperature setpoint for occupied zones
    :param possible_temps: Store the possible temperatures that can be assigned to a zone
    :param **kwargs: Pass additional arguments to the function
    :return: The best zone to assign the visitor group, and the temperature setpoint, None if reject the occupant
    """
    if possible_temps is None:
        current_temp = list()
        for i in range(len(occupants)):
            if occupants[i] != 0:
                current_temp.append(temp[i])
        current_temp.sort()
        current_temp.append(27)

        lb = 20
        new_temps = list()
        for v in current_temp:
            if v - 1 > lb:
                num_new_temp = np.ceil((v - 1 - lb) / 2)
                delta = (v - 1 - lb) / (num_new_temp + 1)
                for i in range(1, int(num_new_temp) + 1):
                    new_temps.append(delta * i + lb)
            lb = v + 1
        possible_temps = sorted(new_temps + current_temp[:-1])

    # Return None if reject the occupant, [zone_id, new_temp] if accept occupant
    candidate = list()
    # First check occupied zones
    for i, zone in enumerate(control_zones):
        if visitor_size + occupants[i] <= capacity[zone] and occupants[i] != 0 and thermal_lb <= temp[i] <= thermal_ub:
            corrected_space = capacity[zone] - occupants[i]
            for j in range(len(control_zones)):
                if occupants[j] == 0:
                    continue
                corrected_space += max(2 - abs(temp[i] - temp[j]), 0) * (capacity[control_zones[j]] - occupants[j]) / 2
            t = temp[i]

            v = [out_night_temp, t, out_day_temp, visitor_size + occupants[i]]
            if use_adj:
                for adj_name in zone_adj[zone]:
                    v.append(temp[control_zones.index(adj_name)])
            e = occupied_models[zone].predict([v])[0]
            v[3] -= visitor_size
            e -= occupied_models[zone].predict([v])[0]
            candidate.append([corrected_space, -e, i, t])

    if not candidate:
        # Check if any unoccupied zone is available, assign large zone first
        temps = list()
        for t in possible_temps:
            if thermal_lb <= t <= thermal_ub:
                overlaps = 0
                for j in range(len(control_zones)):
                    if occupants[j] == 0:
                        continue
                    overlaps += max(2 - abs(t - temp[j]), 0) * (capacity[control_zones[j]] - occupants[j]) / 2
                temps.append([overlaps, t])
        if not temps:
            return None
        t = sorted(temps)[0][1]

        for i, zone in enumerate(control_zones):
            if visitor_size <= capacity[zone] and occupants[i] == 0:
                v = [out_night_temp, t, out_day_temp, visitor_size]
                if use_adj:
                    for adj_name in zone_adj[zone]:
                        v.append(temp[control_zones.index(adj_name)])
                e = occupied_models[zone].predict([v])[0] - empty_models[i]
                candidate.append([capacity[zone], -e, i, t])

        # If still no empty zone, then reject
        if not candidate:
            return None

    candidate.sort()
    # Assign to the zone with most space
    return candidate[-1][2], candidate[-1][3], possible_temps


def online_minlp(control_zones, capacity, zone_adj, use_adj, visitor_size, thermal_lb, thermal_ub,
                 occupied_models, empty_models, out_day_temp, out_night_temp,
                 occupants, temp, occupied_thermal_lb, occupied_thermal_ub, possible_temps=None, **kwargs):
    """
    The online_minlp function is a mixed integer linear programming (MILP) function that optimally assign
    short term occupants in a zone. This is the "Online-MINLP" algorithm in the paper.

    :param control_zones: Specify the zones that can be controlled by the hvac system
    :param capacity: Determine the maximum number of occupants in a zone
    :param zone_adj: Specify the adjacent zones of each zone
    :param use_adj: Indicate whether or not to use adjacent zone temperature as input for the model
    :param visitor_size: Determine the number of short term occupants in a group
    :param thermal_lb: Set the lower bound of temperature based on the short term occupants' preference
    :param thermal_ub: Set the upper bound of temperature based on the short term occupants' preference
    :param occupied_models: Predict the energy consumption of a zone given its temperature setpoint, occupancy, and more
    :param empty_models: Estimate the energy consumption of an empty zone
    :param out_day_temp: The average outdoor temperature during day time
    :param out_night_temp: The average outdoor temperature at night
    :param occupants: The number of occupants in each zone
    :param temp: Pass the current temperature of each zone
    :param occupied_thermal_lb: The lower bound of temperature setpoint for occupied zones
    :param occupied_thermal_ub: The upper bound of temperature setpoint for occupied zones
    :param possible_temps: Store the possible temperatures that can be assigned to a zone
    :param **kwargs: Pass additional arguments to the function
    :return: The best zone to assign the visitor group, and the temperature setpoint, None if reject the occupant
    """
    assignable = list()
    for i, zone in enumerate(control_zones):
        if occupants[i] + visitor_size > capacity[zone]:
            continue
        assignable.append(i)

    zone_base_energy = list()
    for i in assignable:
        zone = control_zones[i]
        if occupants[i] == 0:
            zone_base_energy.append(empty_models[i])
        else:
            v = [out_night_temp, temp[i], out_day_temp, occupants[i]]
            if use_adj:
                for adj_name in zone_adj[zone]:
                    v.append(temp[control_zones.index(adj_name)])
            zone_base_energy.append(occupied_models[zone].predict([v])[0])

    m = gp.Model()
    m.setParam('LogToConsole', 0)
    m.setParam('MIPGap', 1e-5)
    m.setParam('IntFeasTol', 1e-9)
    m.setParam('TimeLimit', 6)

    # Group assignment
    x = list()
    for i in range(len(assignable)):
        x.append(m.addVar(vtype='B', name=f"Assignment zone {assignable[i]}"))
    m.addConstr(quicksum(x[i] for i in range(len(x))) == 1, name=f"Only assign to one zone constraint")

    avg_out_temp_var = m.addVar(lb=out_day_temp, ub=out_day_temp)
    avg_out_night_temp_var = m.addVar(lb=out_night_temp, ub=out_night_temp)

    t = m.addVar(vtype='C', lb=thermal_lb, ub=thermal_ub, name=f"temp set zone")

    t_other = list()
    e = list()
    for i in assignable:
        e.append(m.addVar(vtype='C', name=f"Zone {i} energy true estimation"))
    for i in range(len(control_zones)):
        t_other.append(m.addVar(vtype='C', lb=temp[i], ub=temp[i], name=f"Zone {i} original temp"))

    # Add constraints on matrix X
    for i in range(len(assignable)):
        m.addConstr(t * x[i] <= occupied_thermal_ub[assignable[i]] * x[i], name=f"Zone {i} Thermal constraint (ub)")
        m.addConstr(t * x[i] >= occupied_thermal_lb[assignable[i]] * x[i], name=f"Zone {i} Thermal constraint (ub)")

    # Estimate the energy consumption
    for i in range(len(assignable)):
        temp_o = m.addVar(vtype='C')
        temp_t = m.addVar(vtype='C')
        m.addConstr(temp_o == x[i] * visitor_size + occupants[assignable[i]])
        m.addConstr(temp_t == x[i] * t + (1 - x[i]) * temp[assignable[i]])

        var_list = [avg_out_night_temp_var, temp_t, avg_out_temp_var, temp_o]
        if use_adj:
            for adj_name in zone_adj[control_zones[assignable[i]]]:
                var_list.append(t_other[control_zones.index(adj_name)])
        input_vars = MVar.fromlist([var_list])
        output_vars = MVar.fromlist([[e[i]]])
        add_predictor_constr(m, occupied_models[control_zones[assignable[i]]], input_vars, output_vars)

    # Set objective to minimize the estimated total zone energy consumption
    m.setObjective(quicksum((e[i] - zone_base_energy[i]) * x[i] for i in range(len(x))), GRB.MINIMIZE)
    # m.addCons(objvar >= quicksum(e[i] for i in range(len(control_zones))))

    if kwargs["warm_start"]:
        # print(kwargs["warm_start"])
        ws_i, ws_t = kwargs["warm_start"]
        for i in range(len(assignable)):
            x[i].start = int(assignable[i] == ws_i)
        t.start = ws_t
        m.update()

    try:
        m.optimize()

        if m.SolCount == 0:
            if kwargs["warm_start"]:
                return kwargs["warm_start"]
            return None

        for i in range(len(assignable)):
            if int(np.round(x[i].x)) == 1:
                # print(assignable[i], t.x)
                return assignable[i], t.x
    except gp.GurobiError as e:
        return None


def uniform_number(control_zones, capacity, zone_adj, use_adj, visitor_size, thermal_lb, thermal_ub,
                   occupied_models, empty_models, out_day_temp, out_night_temp,
                   occupants, temp, occupied_thermal_lb, occupied_thermal_ub, zone_thermals=None, **kwargs):
    """
    The uniform_number function is used to determine the best zone to place a visitor in.
    It does this by calculating the number of occupants in each zone, and then choosing
    the one with the lowest existing occupants. It also returns what temperature that zone should be set at.
    The temperature is the average preferred temperature of all occupants in the zone.
    This is the "Uniform-Number" algorithm in paper.

    :param control_zones: Specify the zones that can be controlled by the hvac system
    :param capacity: Determine the maximum number of occupants in a zone
    :param zone_adj: Specify the adjacent zones of each zone
    :param use_adj: Indicate whether or not to use adjacent zone temperature as input for the model
    :param visitor_size: Determine the number of short term occupants in a group
    :param thermal_lb: Set the lower bound of temperature based on the short term occupants' preference
    :param thermal_ub: Set the upper bound of temperature based on the short term occupants' preference
    :param occupied_models: Predict the energy consumption of a zone given its temperature setpoint, occupancy, and more
    :param empty_models: Estimate the energy consumption of an empty zone
    :param out_day_temp: The average outdoor temperature during day time
    :param out_night_temp: The average outdoor temperature at night
    :param occupants: The number of occupants in each zone
    :param temp: Pass the current temperature of each zone
    :param occupied_thermal_lb: The lower bound of temperature setpoint for occupied zones
    :param occupied_thermal_ub: The upper bound of temperature setpoint for occupied zones
    :param zone_thermals: termal preferences of existing occupants
    :param **kwargs: Pass additional arguments to the function
    :return: The best zone to assign the visitor group, and the temperature setpoint, None if reject the occupant
    """
    candidate = list()
    for i, zone in enumerate(control_zones):
        if visitor_size + occupants[i] <= capacity[zone]:
            t = (np.sum(zone_thermals[i]) + (thermal_lb + thermal_ub) * visitor_size / 2) / (
                    occupants[i] + visitor_size)
            v = [out_night_temp, t, out_day_temp, visitor_size + occupants[i]]
            if use_adj:
                for _ in zone_adj[zone]:
                    v.append(t)
            e = occupied_models[zone].predict([v])[0]
            if occupants[i] != 0:
                v[3] -= visitor_size
                e -= occupied_models[zone].predict([v])[0]
            else:
                e -= empty_models[i]
            candidate.append([occupants[i], e, i, t])

    if not candidate:
        return None

    candidate.sort()
    return candidate[0][2], candidate[0][3]


def uniform_ratio(control_zones, capacity, zone_adj, use_adj, visitor_size, thermal_lb, thermal_ub,
                  occupied_models, empty_models, out_day_temp, out_night_temp,
                  occupants, temp, occupied_thermal_lb, occupied_thermal_ub, zone_thermals=None, **kwargs):
    """
    The uniform_ratio function is used to determine the best zone to place a visitor in.
    It does this by calculating the number of occupants in each zone, and then choosing
    the one with the lowest occupied ratio (current number of occupants / capacity).
    It also returns what temperature that zone should be set at.
    The temperature is the average preferred temperature of all occupants in the zone.
    This is the "Uniform-Ratio" algorithm in paper.

    :param control_zones: Specify the zones that can be controlled by the hvac system
    :param capacity: Determine the maximum number of occupants in a zone
    :param zone_adj: Specify the adjacent zones of each zone
    :param use_adj: Indicate whether or not to use adjacent zone temperature as input for the model
    :param visitor_size: Determine the number of short term occupants in a group
    :param thermal_lb: Set the lower bound of temperature based on the short term occupants' preference
    :param thermal_ub: Set the upper bound of temperature based on the short term occupants' preference
    :param occupied_models: Predict the energy consumption of a zone given its temperature setpoint, occupancy, and more
    :param empty_models: Estimate the energy consumption of an empty zone
    :param out_day_temp: The average outdoor temperature during day time
    :param out_night_temp: The average outdoor temperature at night
    :param occupants: The number of occupants in each zone
    :param temp: Pass the current temperature of each zone
    :param occupied_thermal_lb: The lower bound of temperature setpoint for occupied zones
    :param occupied_thermal_ub: The upper bound of temperature setpoint for occupied zones
    :param zone_thermals: termal preferences of existing occupants
    :param **kwargs: Pass additional arguments to the function
    :return: The best zone to assign the visitor group, and the temperature setpoint, None if reject the occupant
    """
    candidate = list()
    for i, zone in enumerate(control_zones):
        if visitor_size + occupants[i] <= capacity[zone]:
            t = (np.sum(zone_thermals[i]) + (thermal_lb + thermal_ub) * visitor_size / 2) / (
                    occupants[i] + visitor_size)
            v = [out_night_temp, t, out_day_temp, visitor_size + occupants[i]]
            if use_adj:
                for _ in zone_adj[zone]:
                    v.append(t)
            e = occupied_models[zone].predict([v])[0]
            if occupants[i] != 0:
                v[3] -= visitor_size
                e -= occupied_models[zone].predict([v])[0]
            else:
                e -= empty_models[i]
            candidate.append([occupants[i] / capacity[zone], e, i, t])

    if not candidate:
        return None

    candidate.sort()
    return candidate[0][2], candidate[0][3]


def random(control_zones, capacity, zone_adj, use_adj, visitor_size, thermal_lb, thermal_ub,
           occupied_models, empty_models, out_day_temp, out_night_temp,
           occupants, temp, occupied_thermal_lb, occupied_thermal_ub, zone_thermals=None, **kwargs):
    """
    The random function is used to determine the best zone to place a visitor in.
    It does this by finding all zones that can accommodate the short term occupants, and then randomly select a
    zone from all valid options. It also returns what temperature that zone should be set at.
    The temperature is the average preferred temperature of all occupants in the zone.
    This is the "Random" algorithm in paper.

    :param control_zones: Specify the zones that can be controlled by the hvac system
    :param capacity: Determine the maximum number of occupants in a zone
    :param zone_adj: Specify the adjacent zones of each zone
    :param use_adj: Indicate whether or not to use adjacent zone temperature as input for the model
    :param visitor_size: Determine the number of short term occupants in a group
    :param thermal_lb: Set the lower bound of temperature based on the short term occupants' preference
    :param thermal_ub: Set the upper bound of temperature based on the short term occupants' preference
    :param occupied_models: Predict the energy consumption of a zone given its temperature setpoint, occupancy, and more
    :param empty_models: Estimate the energy consumption of an empty zone
    :param out_day_temp: The average outdoor temperature during day time
    :param out_night_temp: The average outdoor temperature at night
    :param occupants: The number of occupants in each zone
    :param temp: Pass the current temperature of each zone
    :param occupied_thermal_lb: The lower bound of temperature setpoint for occupied zones
    :param occupied_thermal_ub: The upper bound of temperature setpoint for occupied zones
    :param zone_thermals: termal preferences of existing occupants
    :param **kwargs: Pass additional arguments to the function
    :return: The best zone to assign the visitor group, and the temperature setpoint, None if reject the occupant
    """
    candidates = list()
    for i, zone in enumerate(control_zones):
        if visitor_size + occupants[i] <= capacity[zone]:
            if occupants[i] != 0:
                if thermal_lb <= temp[i] <= thermal_ub:
                    candidates.append([i, temp[i]])
            else:
                candidates.append([i, (thermal_lb + thermal_ub) / 2])

    if candidates:
        selection = candidates[np.random.randint(len(candidates))]
        return selection[0], selection[1]

    return None
