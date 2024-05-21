import numpy as np
import torch
import pandas as pd
import math
from scipy.stats import wasserstein_distance
from utils.dc_dataloader import load_carbon, load_indirect_WUE, load_direct_WUE
from sklearn.preprocessing import MinMaxScaler


# Batch_Size = 128
# trainset_size = int(8759*0.7)
# testset_size = 8759 - trainset_size
def calculate_resource(demand_list, predicted_carbon, lbda):
    res_list=[]
    cost_list =[]
    count=0
    for i in range(len(demand_list)):
        tmp = math.sqrt(lbda*demand_list[i]/predicted_carbon[i])
        res = demand_list[i]+tmp
        res_list.append(res)
        if abs(res-demand_list[i]) < 1e-5:
            cost = res*predicted_carbon[i]
        else:
            cost = res*predicted_carbon[i] + lbda*demand_list[i]/(res-demand_list[i])

        if torch.isnan(cost):
            count = count+1
        cost_list.append(cost)
    return res_list, cost_list

def get_total_demand(ev_data):
    demands = ev_data['total_power']  # .drop("Unnamed: 0", axis=1)
    return demands

def get_total_charging_per_unit(ev_data):
    ev_data['duration'] = pd.to_timedelta(ev_data['duration'])
    duration_seconds = ev_data['duration'].dt.total_seconds().astype(int).to_numpy()
    charging_per_second = ev_data['total_power'].to_numpy()/duration_seconds
    return charging_per_second, duration_seconds

def read_demand_data(ev_data, N, trainset_size):
    demands = ev_data['total_power']
    demands_train = demands[:trainset_size]  # trainset size
    demands_test = demands[trainset_size:]  # testset size

    demands_groups_train = np.array_split(demands_train, N)
    demands_groups_test = np.array_split(demands_test, N)
    return demands_groups_train, demands_groups_test

def read_charging_per_unit_data(ev_data, N, trainset_size):
    charging_per_second, duration_seconds = get_total_charging_per_unit(ev_data)
    charging_per_unit_train = charging_per_second[:trainset_size]  # trainset size
    charging_per_unit_test = charging_per_second[trainset_size:]  # testset size
    assert len(charging_per_unit_train)+len(charging_per_unit_test) == len(ev_data)
    charging_per_unit_groups_train = np.array_split(charging_per_unit_train, N)
    charging_per_unit_groups_test = np.array_split(charging_per_unit_test, N)
    return charging_per_unit_groups_train, charging_per_unit_groups_test

def cal_wass_distance(data):
    n = len(data)
    wass_distances = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            distance = wasserstein_distance(data[i], data[j])
            wass_distances[i, j] = distance
            wass_distances[j, i] = distance

    return np.min(wass_distances[wass_distances > 0]), np.max(wass_distances)



