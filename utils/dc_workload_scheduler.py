import numpy as np
import torch
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import wasserstein_distance


def calculate_resource(demand_list, predicted_carbon, lbda):
    res_list=[]
    cost_list =[]
    for i in range(len(demand_list)):
        tmp = math.sqrt(lbda*demand_list[i]/predicted_carbon[i])
        res = demand_list[i]+tmp
        res_list.append(res)
        if abs(res-demand_list[i]) < 1e-5:
            cost = res*predicted_carbon[i]
        else:
            cost = res*predicted_carbon[i] + lbda*demand_list[i]/(res-demand_list[i])
        cost_list.append(cost)

    return res_list, cost_list

def calculate_cost_in_cuda(demand_list, action, carbon, lbda):
    if len(demand_list) != len(carbon):
        smaller_len = len(demand_list) if len(demand_list) < len(carbon) else len(carbon)
        demand_list = demand_list[:smaller_len]
        carbon = carbon[:smaller_len]

    demands = torch.tensor(demand_list, dtype=torch.float64).to('cuda').unsqueeze(1)
    cost = action*carbon + lbda*demands/(action-demands)
    return cost

def calculate_action_in_cuda(demand_list, carbon, lbda):
    if len(demand_list) != len(carbon):
        smaller_len = len(demand_list) if len(demand_list) < len(carbon) else len(carbon)
        demand_list = demand_list[:smaller_len]
        carbon = carbon[:smaller_len]

    demands = torch.tensor(demand_list, dtype=torch.float64).clone().to('cuda').unsqueeze(1)
    div = torch.div(lbda*demands, carbon)
    tmp = torch.sqrt(div)
    action = demands + tmp  # action
    if torch.isinf(action).any():
        inf_mask = torch.isinf(action)
        indix = torch.where(inf_mask)
        print("index", indix, "demands:", demands[indix], "carbon:", carbon[indix])

    return action

def split_group(num_groups, demand_list, predicted_carbon_list):
    lambda_list=[0.0001]
    group_size = int(len(demand_list)/num_groups)
    for i in range(num_groups):
        cur = lambda_list[-1]*10
        lambda_list.append(cur)

        carbon_groups = [predicted_carbon_list[i:i+group_size] for i in range(0, len(predicted_carbon_list), group_size)]
        demand_groups = [demand_list[i:i+group_size] for i in range(0, len(demand_list), group_size)]

    print("lambda_list: ", lambda_list, "group size: ", len(carbon_groups[0]), len(demand_groups[0]))
    return carbon_groups, demand_groups, lambda_list

def preprocess_workload(demand_path):
    df = pd.read_csv(demand_path)[1:]  # .drop("Unnamed: 0", axis=1)
    df_demand = df.reset_index(drop=True)
    df_demand['Time'] = pd.to_datetime(df_demand['Time'])

    # Group the DataFrame by year, month, day, hour, and minute
    grouped_df = df_demand.groupby([df_demand['Time'].dt.year, df_demand['Time'].dt.month, df_demand['Time'].dt.day,
                                    df_demand['Time'].dt.hour]).sum()

    scaler = MinMaxScaler(feature_range=(0, 10))
    df_scaled = pd.DataFrame(scaler.fit_transform(grouped_df))
    return df_scaled


def load_demand(demand_path):
    grouped_df = preprocess_workload(demand_path)
    demand_values = grouped_df.iloc[6:2891] # testset size
    demand_list = demand_values[0].values

    return demand_list

def read_demand_data(demand_path, N):
    grouped_df = preprocess_workload(demand_path)
    demand_train= grouped_df.iloc[-5867:-5][0].values  # trainset size
    demand_test = grouped_df.iloc[6:2891][0].values  # testset size
    groups_train = np.array_split(demand_train, N)
    groups_test = np.array_split(demand_test, N)
    return groups_train, groups_test

def cal_wass_distance(data):
    n = len(data)
    wass_distances = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            distance = wasserstein_distance(data[i], data[j])
            wass_distances[i, j] = distance
            wass_distances[j, i] = distance

    return np.min(wass_distances[wass_distances > 0]), np.max(wass_distances)














