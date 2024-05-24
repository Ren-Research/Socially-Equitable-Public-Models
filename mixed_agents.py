from utils.dc_dataloader import load_carbon
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from utils.dc_workload_scheduler import calculate_action_in_cuda, calculate_cost_in_cuda
from utils.ev_scheduler import read_charging_per_unit_data
from backbones.transformer_multistep import TransAm
from utils.mix_agents_helper import read_dc_data, read_iphone_data, read_ev_data, expand_iphone_dataF, calculate_mean
from utils.add_params import add_general_args
from argparse import ArgumentParser


def create_sequences(data, seq_length, num_time_steps_to_predict):
    xs = []
    ys = []

    for i in range(len(data) - seq_length - num_time_steps_to_predict+1):
        x = data[i:(i+seq_length)]
        y = data[(i + seq_length): (i + seq_length + num_time_steps_to_predict)]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

def create_sequences_dc(data, seq_length):
    xs = []
    ys = []

    for i in range(len(data) - seq_length - 1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

def split_train_test(data, train_ratio):
    train_size = int(len(data) * train_ratio)
    train_data, test_data = data[0:train_size, :], data[train_size:len(data), :]
    return train_data, test_data

def read_carbon_data():
    loc_name_list = ["4S2_Oregan_NW", "HND_Nevada_CAL", "JYO_virginia_PJM", "JWY_Texas_ERCO"]
    loc_name = loc_name_list[1]
    fuel_mix_path = "../data/fuelmix/{}_year_2022.csv".format(loc_name.split("_")[-1])
    dc_loc = loc_name.split("_")[1]
    carbon_curve = load_carbon(fuel_mix_path, dc_loc)
    data = carbon_curve.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 10))
    normed_data = scaler.fit_transform(data)
    return normed_data

def calculate_dc_cost(output, labels, lbda, demands, q):
    labels[labels <= 0] = 0.01
    output[output <= 0] = 0.1
    output = output.mean(dim=1).reshape(-1, 1)
    labels = labels.mean(dim=1).reshape(-1, 1)
    true_action = calculate_action_in_cuda(demands, labels, lbda)
    true_cost = calculate_cost_in_cuda(demands, true_action, labels, lbda)

    pred_action = calculate_action_in_cuda(demands, output.unsqueeze(1), lbda)
    pred_cost = calculate_cost_in_cuda(demands, pred_action, labels, lbda)
    loss = torch.pow(torch.mean(pred_cost - true_cost), q)

    return loss

def calculate_ev_cost(output, labels, charge_per_seconds, demands, q):
    output[output <= 0] = 0.1
    demands = demands.tolist()
    k_list = [int(demands[i] / (charge_per_seconds[i] * 3600)) for i in range(min(len(demands), len(charge_per_seconds)))]
    labels[labels <= 0] = 0.01

    pred_costs, true_costs = [], []
    k_list = [min(x, 12) for x in k_list]
    k_list = [max(x, 1) for x in k_list]
    for i in range(0, min(len(output), len(k_list))):
        pred_action, _ = torch.topk(output[i], k_list[i], dim=0, largest=False)
        pred_cost = torch.sum(pred_action)
        pred_costs.append(pred_cost)

        true_action, _ = torch.topk(labels[i], k_list[i], dim=0, largest=False)
        true_cost = torch.sum(true_action)
        true_costs.append(true_cost)

    losses = torch.stack([(t1 - t2) for t1, t2 in zip(pred_costs, true_costs)])
    loss = torch.pow(torch.mean(losses, dim=0), q)

    return loss

def calculate_ipone_cost(output, labels, demands, q):
    output[output <= 0] = 0.1
    demands = demands.tolist()
    k_list = np.random.randint(1, 11, size=len(demands))
    labels[labels <= 0] = 0.01

    pred_costs, true_costs = [], []
    for i in range(0, len(output)):
        pred_action, _ = torch.topk(output[i], k_list[i], dim=0, largest=False)
        pred_cost = torch.sum(pred_action)
        pred_costs.append(pred_cost)

        true_action, _ = torch.topk(labels[i], k_list[i], dim=0, largest=False)
        true_cost = torch.sum(true_action)
        true_costs.append(true_cost)

    losses = torch.stack([(t1 - t2) for t1, t2 in zip(pred_costs, true_costs)])
    loss = torch.pow(torch.mean(losses, dim=0), q)

    return loss

def test(test_X, test_y, model, cuda_size):
    test_data = TensorDataset(test_X, test_y)
    test_loader = DataLoader(test_data, batch_size=cuda_size, shuffle=False) # adjust cuda_size value to fit your GPU memory
    predicted = []
    model.eval()
    with torch.no_grad():
        for test_seq, test_label in test_loader:
            test_seq, test_label = test_seq.to('cuda'), test_label.to('cuda')
            output = model(test_seq)
            predicted.append(output)

    return [item for sublist in predicted for item in sublist]

def calculate_cost_in_cuda_ev(demands, charge_per_seconds, carbon):
    demands = demands.tolist()
    k_list = [int(demands[i] / (charge_per_seconds[i] * 3600)) for i in
              range(min(len(demands), len(charge_per_seconds)))]
    carbon[carbon <= 0] = 0.01

    k_list = [min(x, 12) for x in k_list]
    k_list = [max(x, 1) for x in k_list]
    assert all(element <= 12 for element in k_list)

    costs = []
    for i in range(0, len(carbon)):
        action, _ = torch.topk(carbon[i], k_list[i], dim=0, largest=False)
        cost = torch.sum(action)
        costs.append(cost)

    return costs

def calculate_cost_in_cuda_iphone(output, labels, demands):
    k_list = np.random.randint(1, 11, size=len(demands))
    labels[labels <= 0] = 0.01

    pred_costs, true_costs = [], []
    for i in range(0, len(output)):
        pred_action, _ = torch.topk(output[i], k_list[i], dim=0, largest=False)
        pred_cost = torch.sum(pred_action)
        pred_costs.append(pred_cost)

        true_action, _ = torch.topk(labels[i], k_list[i], dim=0, largest=False)
        true_cost = torch.sum(true_action)
        true_costs.append(true_cost)
    return pred_costs, true_costs

def get_carbon_from_PM(carbon_test_data, model):
    test_X, test_y = create_sequences(carbon_test_data, seq_length, 12)
    test_y[test_y <= 0] = 0.01
    test_X = torch.from_numpy(test_X)
    test_y = torch.from_numpy(test_y)

    predicted_carbon = test(test_X, test_y, model, cuda_size=128)
    predicted_carbon = torch.stack(predicted_carbon, dim=0)
    predicted_carbon[predicted_carbon <= 0] = 0.01
    test_y = torch.stack(tuple(test_y), dim=0)

    criterion = nn.MSELoss()
    mse = criterion(predicted_carbon, test_y.cuda())

    return torch.mean(predicted_carbon, dim=0), torch.mean(test_y, dim=0), mse


if __name__ == '__main__':
    parser = ArgumentParser(description='datacenter-app-build-public-models')
    args = add_general_args(parser)

    seq_length = 12
    num_time_steps_to_predict = 12
    N = 3 # Agents: Data center, EVs, and iPhones
    ev_data = read_ev_data()
    iphone_data = read_iphone_data()
    dc_data = read_dc_data()

    normed_carbon_data = read_carbon_data()

    model = TransAm().double().to('cuda')
    num_epochs = args.n_epochs
    learning_rate = args.lr
    batch_size = args.batch_size
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    val_data = []
    carbon_train_data, carbon_test_data = split_train_test(normed_carbon_data, train_ratio=args.train_ratio) # train_ratio = 0.67
    carbon_train_data = carbon_train_data.astype(np.float64)
    carbon_test_data = carbon_test_data.astype(np.float64)

    train_X, train_y = create_sequences(carbon_train_data, seq_length, num_time_steps_to_predict)

    train_X = torch.from_numpy(train_X)
    train_y = torch.from_numpy(train_y)

    train_dataset = TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    sub_groups_charging_per_sec_train, sub_groups_charging_per_sec_test = read_charging_per_unit_data(ev_data, N, int(len(normed_carbon_data)*0.7))

    model.train()
    w_dc, w_ev, w_ip = 10, 1, 0.2

    if args.training:
        for epoch in range(0, num_epochs):
            for batch_idx, (carbon_data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                seq = carbon_data.to('cuda')
                labels = target.to('cuda')
                output = model(seq.double())  # carbon emission
                if args.baseline:
                    mse_loss = nn.MSELoss()
                    loss = mse_loss(output, labels)
                    loss = loss/N
                else:
                    loss = 0
                    q = args.q_idx
                    for n in range(N):
                        if n == 0:
                            dc_demands = dc_data.tolist()
                            cost_b_n = w_dc*calculate_dc_cost(output, labels, 2, dc_demands, q)
                        elif n == 1:
                            charge_per_unit = sub_groups_charging_per_sec_train[n]
                            demands = ev_data['total_power']
                            cost_b_n = w_ev*calculate_ev_cost(output, labels, charge_per_unit, demands, q)
                        else:
                            expanded_iphone_data = expand_iphone_dataF(iphone_data, batch_size)
                            demands = expanded_iphone_data['charging_demands']
                            cost_b_n = w_ip*calculate_ipone_cost(output, labels, demands, q)

                        loss = loss + cost_b_n
                    loss = torch.pow(loss, 1 / q) / N

                loss.backward(retain_graph=True)

                optimizer.step()
                scheduler.step()

            if epoch % 10 == 0:
                print(f'Iter: {epoch}, Loss: {loss.item()}')

    # Uncomment below and modify the path when training your own public models to save
    # path = "./trained_models/my_model.pth"
    # torch.save(model.state_dict(), path)

    print("---start testing---")
    if not args.training:
        model.load_state_dict(torch.load(args.model_path))

    true_cost_list_groups = []
    pred_cost_list_groups = []

    model.eval()

    inf_carbon_pred, inf_carbon_true, mse_metric = get_carbon_from_PM(carbon_test_data, model)
    ip_test_syn = 5

    for n in range(N):
        expanded_inf_carbon_pred = inf_carbon_pred.unsqueeze(0).repeat(len(carbon_test_data), 1, 1).cuda()
        expanded_inf_carbon_true = inf_carbon_true.unsqueeze(0).repeat(len(carbon_test_data), 1, 1).cuda()
        if n == 0:
            dc_demands = dc_data.iloc[-len(carbon_test_data):-1].tolist()
            pred = expanded_inf_carbon_pred.mean(dim=1).reshape(-1, 1)
            labels = expanded_inf_carbon_true.mean(dim=1).reshape(-1, 1)
            pred_action_list = calculate_action_in_cuda(dc_demands, pred, 2)
            pred_cost = calculate_cost_in_cuda(dc_demands, pred_action_list, labels, 2)
            true_action_list = calculate_action_in_cuda(dc_demands, labels.mean(dim=1).reshape(-1, 1), 2)
            true_cost = calculate_cost_in_cuda(dc_demands, true_action_list, labels.mean(dim=1).reshape(-1, 1), 2)
            _, pred_cost_list = calculate_mean(pred_cost)
            _, true_cost_list = calculate_mean(true_cost)
            pred_cost_list, true_cost_list = [w_dc * x for x in pred_cost_list], [w_dc * x for x in true_cost_list]
            pred_cost_list_groups.append(pred_cost_list)
            true_cost_list_groups.append(true_cost_list)

        elif n == 1:
            ev_test_demands = ev_data['total_power'][:len(carbon_test_data)]
            charge_per_unit = [item for sublist in sub_groups_charging_per_sec_test for item in sublist][:len(carbon_test_data)]
            pred_cost = calculate_cost_in_cuda_ev(ev_test_demands, charge_per_unit, expanded_inf_carbon_pred)
            true_cost = calculate_cost_in_cuda_ev(ev_test_demands, charge_per_unit, expanded_inf_carbon_true)
            _, pred_cost_list = calculate_mean(pred_cost)
            _, true_cost_list = calculate_mean(true_cost)
            pred_cost_list, true_cost_list = [w_ev*x for x in pred_cost_list], [w_ev*x for x in true_cost_list]
            pred_cost_list_groups.append(pred_cost_list)
            true_cost_list_groups.append(true_cost_list)

        else:
            ip_test_demands = [ip_test_syn]*len(carbon_test_data)
            pred_cost, true_costs = calculate_cost_in_cuda_iphone(expanded_inf_carbon_pred, expanded_inf_carbon_true, ip_test_demands)
            _, pred_cost_list = calculate_mean(pred_cost)
            _, true_cost_list = calculate_mean(true_cost)
            pred_cost_list, true_cost_list = [np.abs(w_ip * x) for x in pred_cost_list], [np.abs(w_ip * x) for x in true_cost_list]

            pred_cost_list_groups.append(pred_cost_list)
            true_cost_list_groups.append(true_cost_list)

    pred_means = []
    diffs = []

    for n in range(0, len(pred_cost_list_groups)):
        diffs.append(abs(np.mean(pred_cost_list_groups[n]) - np.mean(true_cost_list_groups[n])))

    print("difference variance: ", np.var(diffs), "means of diffs", np.mean(diffs))



