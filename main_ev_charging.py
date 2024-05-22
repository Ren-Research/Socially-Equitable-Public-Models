import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime
from argparse import ArgumentParser
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from backbones.transformer_multistep import TransAm
from utils.ev_scheduler import read_demand_data, read_charging_per_unit_data, cal_wass_distance
from utils.add_params import add_general_args
from utils.dc_dataloader import load_carbon, load_indirect_WUE, load_direct_WUE

def read_data_water():
    loc_name_list = ["4S2_Oregan_NW", "HND_Nevada_CAL", "JYO_virginia_PJM", "JWY_Texas_ERCO"]
    loc_name = loc_name_list[1]
    fuel_mix_path = "./data/fuelmix/{}_year_2022.csv".format(loc_name.split("_")[-1])
    weather_path = "./data/weather/{}.csv".format(loc_name)
    dc_loc = loc_name.split("_")[1]

    indirectWue = load_indirect_WUE(fuel_mix_path, dc_loc) * 1.1
    directWue = load_direct_WUE(weather_path)

    Wue = directWue + indirectWue
    data = Wue.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 10))
    normed_data = scaler.fit_transform(data)
    return normed_data

def read_data_elec_price():
    elec_prices = pd.read_csv("./data/elec_prices_hrly.csv")
    filtered_rows = elec_prices[(elec_prices['hasp_price_per_mwh'] > 10) & (elec_prices['hasp_price_per_mwh'] < 100)]
    prices = np.array(filtered_rows['hasp_price_per_mwh']).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 10))
    normed_data = scaler.fit_transform(prices)
    return normed_data

def read_data_carbon():
    loc_name_list = ["4S2_Oregan_NW", "HND_Nevada_CAL", "JYO_virginia_PJM", "JWY_Texas_ERCO"]
    loc_name = loc_name_list[1]
    fuel_mix_path = "./data/fuelmix/{}_year_2022.csv".format(loc_name.split("_")[-1])
    dc_loc = loc_name.split("_")[1]
    carbon_curve = load_carbon(fuel_mix_path, dc_loc)
    data = carbon_curve.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 10))
    normed_data = scaler.fit_transform(data)
    return normed_data

def split_train_test(data, train_ratio):
    train_size = int(len(data) * train_ratio)
    train_data, test_data = data[0:train_size, :], data[train_size:len(data), :]
    return train_data, test_data

def create_sequences(data, seq_length, num_time_steps_to_predict):
    xs = []
    ys = []

    for i in range(len(data) - seq_length - num_time_steps_to_predict+1):
        x = data[i:(i+seq_length)]
        y = data[(i + seq_length): (i + seq_length + num_time_steps_to_predict)]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

def test(test_X, test_y, model, cuda_size):
    test_data = TensorDataset(test_X, test_y)
    test_loader = DataLoader(test_data, batch_size=cuda_size, shuffle=False)
    predicted = []
    model.eval()
    with torch.no_grad():
        for test_seq, test_label in test_loader:
            test_seq, test_label = test_seq.to('cuda'), test_label.to('cuda')
            output = model(test_seq)
            predicted.append(output)

    return [item for sublist in predicted for item in sublist]


def convert_to_str_days(data_str):
    base_date = datetime.strptime("2022-01-01T01", '%Y-%m-%dT%H')
    date_1 = datetime.strptime(data_str, '%Y-%m-%dT%H')
    delta_t = date_1 - base_date

    return delta_t.days

def calculate_mean(cost_list):
    array = np.array([tensor.cpu().numpy() for tensor in cost_list])
    normalized_array = (array - np.min(array)) / (np.max(array) - np.min(array))

    return normalized_array, array

def calculate_K(demands, charge_per_seconds, bound = 12):
    demands = demands.tolist()
    k_list = [int(demands[i]/(charge_per_seconds[i]*3600)) for i in range(len(demands))]
    if any(element >= bound for element in k_list):
        k_list = [12 if element >= bound else element for element in k_list]
    return k_list

def get_charging_time_window(charging_time_window):
    df = charging_time_window
    res = df.groupby('HOUSEID').agg({'STRTTIME': 'min', 'ENDTIME': 'max'}).reset_index()
    formatted_strtTime = [f'{element:04}' for element in res['STRTTIME']]
    formatted_endTime = [f'{element:04}' for element in res['ENDTIME']]
    st_hr = [int(element[:2]) for element in formatted_strtTime]
    et_hr = [int(element[:2]) for element in formatted_endTime]
    dur_hr = [(et_hr[i]-st_hr[i]+1) for i in range(len(et_hr))]
    trips_larger_than_zero = [element for element in dur_hr if element > 0]
    charging_dur = [(24 - ele) for ele in trips_larger_than_zero]
    res_charging_dur = [element for element in charging_dur if element ==12]
    stacked_list = [list(sublist) for sublist in zip(st_hr, et_hr, res_charging_dur)]
    return res_charging_dur, stacked_list

def cost_group_loss(output, labels, charge_per_seconds, demands, q, time_window):
    k_list = calculate_K(demands, charge_per_seconds, bound = time_window[0])
    labels_modified = torch.where(labels <= 0, torch.tensor(0.01, dtype=labels.dtype, device=labels.device), labels)

    pred_costs, true_costs= [], []
    for i in range(0, min(len(output), len(k_list))):
        pred_action, _ = torch.topk(output[i], k_list[i], dim=0, largest=False)
        pred_cost = torch.sum(pred_action)
        pred_costs.append(pred_cost)

        true_action, _ = torch.topk(labels_modified[i], k_list[i], dim=0, largest=False)
        true_cost = torch.sum(true_action)
        true_costs.append(true_cost)

    losses = torch.stack([(t1 - t2) for t1, t2 in zip(pred_costs, true_costs)])
    loss = torch.pow(torch.mean(losses, dim=0), q)

    return loss

def combine_water_and_carbon_and_price(carbon, water, price, gamma, eta):
    min_len = min(len(carbon), len(price))
    carbon_m, water_m, price_m = carbon[:min_len], water[:min_len], price[:min_len]
    return carbon_m + gamma*water_m + eta*price_m

def preprocess_ev_charging_data(total_ev_data):
    total_ev_data['start_time'] = pd.to_datetime(total_ev_data['start_time'])
    total_ev_data['done_charge'] = pd.to_datetime(total_ev_data['done_charge'])
    total_ev_data['duration'] = total_ev_data['done_charge'] - total_ev_data['start_time']
    candidate_ev_data = total_ev_data[total_ev_data['start_time'] < total_ev_data['done_charge']]
    candidate_ev_data['duration'] = pd.to_timedelta(candidate_ev_data['duration'])
    duration_seconds = candidate_ev_data['duration'].dt.total_seconds().astype(int).to_numpy()
    candidate_index = np.where((duration_seconds/3600 >= 3) & (duration_seconds/3600 <= 12))[0]
    return candidate_ev_data.iloc[candidate_index]

def calculate_cost_in_cuda(demands, charge_per_seconds, time_window, cwp):
    k_list = calculate_K(demands, charge_per_seconds, bound=time_window[0])
    cwp[cwp <= 0] = 0.01
    costs = []
    for i in range(0, len(cwp)):
        action, _ = torch.topk(cwp[i], k_list[i], dim=0, largest=False)
        cost = torch.sum(action)
        costs.append(cost)

    return costs

def get_cwp_from_PM(test_data):
    test_X, test_y = create_sequences(test_data, seq_length, num_time_steps_to_predict)
    test_y[test_y <= 0] = 0.01
    test_X = torch.from_numpy(test_X)
    test_y = torch.from_numpy(test_y)

    predicted_cwp = test(test_X, test_y, model, cuda_size=128)
    stacked_predicted_cwp = torch.stack(predicted_cwp, dim=0)
    criterion = nn.MSELoss()
    mse = criterion(stacked_predicted_cwp, test_y.cuda())

    return torch.mean(stacked_predicted_cwp, dim=0), torch.mean(test_y, dim=0), torch.mean(mse, dim=0)


if __name__ == '__main__':
    parser = ArgumentParser(description='datacenter-app-build-public-backbones')
    args = add_general_args(parser)

    seq_length = 12
    num_time_steps_to_predict = 12
    N = 70
    trainset_ratio = 0.7
    Batch_Size = args.batch_size

    carbon_data = read_data_carbon()
    water_data = read_data_water()
    price_data = read_data_elec_price()
    cwp_data = combine_water_and_carbon_and_price(carbon_data, water_data, price_data, gamma=1, eta=1)

    trainset_size = int(len(cwp_data)*trainset_ratio)
    cwp_data_train = cwp_data[:trainset_size]
    cwp_data_test = cwp_data[trainset_size:]

    train_X, train_y = create_sequences(cwp_data_train, seq_length, num_time_steps_to_predict)
    train_X = torch.from_numpy(train_X)
    train_y = torch.from_numpy(train_y)

    train_dataset = TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, batch_size=Batch_Size, shuffle=False)

    total_ev_data = pd.read_csv("./data/caltech_ev_dataset_detail.csv")
    candidate_ev_data = preprocess_ev_charging_data(total_ev_data)
    dataset_size = min(len(cwp_data), len(total_ev_data))
    candidate_ev_data = candidate_ev_data[:dataset_size]

    if args.diff_group_dist:
        sub_groups_demand_train, sub_groups_demand_test = np.load('./data/ev_demands_sub_groups_train_dist.npy', allow_pickle=True), \
                                                      np.load('./data/ev_demands_sub_groups_test_dist.npy', allow_pickle=True)
    else:
        sub_groups_demand_train, sub_groups_demand_test = read_demand_data(candidate_ev_data, N, trainset_size)

    wass_dist_min, wass_dist_max = cal_wass_distance(sub_groups_demand_test)
    print("Wass distance of demands between groups: [", wass_dist_min, ", ", wass_dist_max,
          "]")

    sub_groups_charging_per_sec_train, sub_groups_charging_per_sec_test = read_charging_per_unit_data(
        candidate_ev_data, N,  int(len(cwp_data)*trainset_ratio))

    for n in range(0, N):
        assert len(sub_groups_charging_per_sec_test[n]) == len(sub_groups_demand_test[n])

    charging_time_window = pd.read_csv("./data/trippub.csv")
    time_window, _ = get_charging_time_window(charging_time_window)

    model = TransAm().double().to('cuda')

    baseline = args.baseline
    beta = 0.0

    # start training
    if args.training:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        for epoch in range(0, args.n_epochs):
            for batch_idx, (cwp, target) in enumerate(train_loader):
                optimizer.zero_grad()
                seq = cwp.to('cuda')
                labels = target.to('cuda')
                output = model(seq.double()) # [batch_size, 12, 1]
                if args.baseline:
                    mse_loss = nn.MSELoss()
                    loss = mse_loss(output, labels)
                    loss = loss/N
                else:
                    loss = 0
                    for n in range(N):
                        demands = sub_groups_demand_train[n]
                        charge_per_unit = sub_groups_charging_per_sec_train[n]
                        loss_b_n = cost_group_loss(output, labels, charge_per_unit, demands, args.q_idx, time_window)
                        if beta != 0:
                            mse_l = nn.MSELoss()
                            mse_l_b_n = mse_l(output, labels)
                            loss_b_n = loss_b_n * (1-beta) + beta*mse_l_b_n
                        loss = loss + loss_b_n
                    loss = torch.pow(loss, 1/args.q_idx)/N

                loss.backward(retain_graph=True)
                optimizer.step()
                scheduler.step()

            if epoch % 10 == 0:
                if args.baseline:
                    print(f'Iter: {epoch}, MSE Loss: {loss.item()}')
                else:
                    print(f'Iter: {epoch}, Cost: {loss.item()}')

    # Uncomment below and modify the path when training your own public models to save
    # path = "./trained_models/my_model.pth"
    # torch.save(model.state_dict(), path)

    # start testing
    print("---Start Evaluation---")
    true_cost_list_groups = []
    pred_cost_list_groups = []

    if not args.training:
        model.load_state_dict(torch.load(args.model_path))

    mse_list = []
    model.eval()
    inf_cwp_pred, inf_cwp_true, mse_metric = get_cwp_from_PM(cwp_data_test)

    for n in range(0, N):
        test_demands = sub_groups_demand_test[n]
        charge_per_unit = sub_groups_charging_per_sec_test[n]

        expanded_inf_cwp_pred = inf_cwp_pred.repeat(len(test_demands), 1, 1)
        expanded_inf_cwp_true = inf_cwp_true.repeat(len(test_demands), 1, 1).to('cuda')

        pred_cost = calculate_cost_in_cuda(test_demands, charge_per_unit, time_window, expanded_inf_cwp_pred)
        true_cost = calculate_cost_in_cuda(test_demands, charge_per_unit, time_window, expanded_inf_cwp_true)

        _, pred_cost_list = calculate_mean(pred_cost)
        _, true_cost_list = calculate_mean(true_cost)
        arr = pred_cost_list - true_cost_list

        pred_cost_list_groups.append(pred_cost_list)
        true_cost_list_groups.append(true_cost_list)

    pred_means = []
    diffs = []

    for n in range(0, len(pred_cost_list_groups)):
        pred_means.append(np.mean(pred_cost_list_groups[n]))
        diffs.append(abs(np.mean(pred_cost_list_groups[n]) - np.mean(true_cost_list_groups[n])))

    print("Variance: ", np.var(diffs), "Means", np.mean(diffs))















