import torch
import torch.nn as nn
import numpy as np
from backbones.lstm import LSTM
from torch.utils.data import DataLoader, TensorDataset
from utils.dc_dataloader import read_data
from utils.dc_workload_scheduler import calculate_action_in_cuda, calculate_cost_in_cuda, read_demand_data, cal_wass_distance
from argparse import ArgumentParser
from utils.add_params import add_general_args

def split_train_test(data, train_ratio):
    train_size = int(len(data) * train_ratio)
    train_data, test_data = data[0:train_size, :], data[train_size:len(data), :]
    return train_data, test_data

def create_sequences(data, seq_length):
    xs = []
    ys = []

    for i in range(len(data) - seq_length - 1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

def test(test_X, test_y, model):
    cuda_size = 128  # adjust this value to fit your GPU memory
    test_data = TensorDataset(test_X, test_y)
    test_loader = DataLoader(test_data, batch_size=cuda_size, shuffle=False)
    predicted = []
    # Evaluate the model
    # model.eval()
    with torch.no_grad():
        for test_seq, test_label in test_loader:
            test_seq, test_label = test_seq.to('cuda'), test_label.to('cuda')
            output = model(test_seq)
            predicted.append(output)

    return [item for sublist in predicted for item in sublist]

def get_carbon_from_PM(carbon_test_data):
    test_X, test_y = create_sequences(carbon_test_data, seq_length)

    test_y[test_y <= 0] = 0.01
    test_X = torch.from_numpy(test_X)
    test_y = torch.from_numpy(test_y)

    predicted_carbon = test(test_X, test_y, model)

    criterion = nn.MSELoss()
    mse = criterion(torch.tensor(predicted_carbon).view(-1, 1), test_y)

    return torch.mean(torch.stack(predicted_carbon, dim=0)), torch.mean(test_y, dim=0), torch.mean(mse, dim=0)

def calculate_mean(cost_list):
    array = np.array([tensor.cpu().numpy() for tensor in cost_list])
    normalized_array = (array - np.min(array)) / (np.max(array) - np.min(array))

    return normalized_array, array


def cost_group_loss(output, labels, lbda, demands, q):
    labels_modified = torch.where(labels <= 0, torch.tensor(0.01, dtype=labels.dtype, device=labels.device), labels)

    true_action = calculate_action_in_cuda(demands, labels_modified, lbda)
    true_cost = calculate_cost_in_cuda(demands, true_action, labels_modified, lbda)

    pred_action = calculate_action_in_cuda(demands, output.unsqueeze(1), lbda)
    pred_cost = calculate_cost_in_cuda(demands, pred_action, labels, lbda)

    loss = torch.pow(torch.mean(pred_cost - true_cost), q)

    if torch.isnan(loss):
        print("pred:", pred_cost, "true:", true_cost)
    return loss


if __name__ == '__main__':
    parser = ArgumentParser(description='datacenter-app-build-public-backbones')
    args = add_general_args(parser)

    N = 50
    seq_length = 12

    if args.diff_lambda:
        lambda_list = [2 * i for i in range(1, 51)]
    else:
        lambda_list = [2.0]*N

    normed_carbon_data = read_data() # carbon_data
    if not args.diff_group_dist:
        sub_groups_demand_train, sub_groups_demand_test = read_demand_data("./data/azure_total_demand.csv", N)
    else:
        sub_groups_demand_train, sub_groups_demand_test = np.load('./data/dc_demands_sub_groups_train_dist.npy', allow_pickle=True), \
                                                          np.load('./data/dc_demands_sub_groups_test_dist.npy', allow_pickle=True)

    # split train and test subgroups
    val_data = []
    carbon_train_data, carbon_test_data = split_train_test(normed_carbon_data, train_ratio=args.train_ratio)
    carbon_train_data = carbon_train_data.astype(np.float64)
    carbon_test_data = carbon_test_data.astype(np.float64)

    train_X, train_y = create_sequences(carbon_train_data, seq_length)

    train_X = torch.from_numpy(train_X)
    train_y = torch.from_numpy(train_y)

    train_dataset = TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    # start training
    model = LSTM().double().to('cuda')

    if args.training:
        print("---Start Training---\n")
        model.train()
        print("baseline: ", args.baseline)
        torch.autograd.set_detect_anomaly(True)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(0, args.n_epochs):
            for batch_idx, (carbon_data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                seq = carbon_data.to('cuda')
                labels = target.to('cuda')
                output = model(seq.double())
                if args.baseline:
                    mse_loss = nn.MSELoss()
                    loss = mse_loss(output, labels)
                else:
                    loss = 0
                    for n in range(N):
                        demands = sub_groups_demand_train[n]
                        loss_b_n = cost_group_loss(output, labels, lambda_list[n], demands, args.q_idx)
                        loss = loss + loss_b_n

                    loss = torch.pow(loss, 1 / args.q_idx) / N
                loss.backward(retain_graph=True)
                optimizer.step()

            if epoch % 10 == 0:
                if args.baseline:
                    print(f'Iter: {epoch}, MSE Loss: {loss.item()}')
                else:
                    print(f'Iter: {epoch}, Cost: {loss.item()}')

    # Uncomment below and modify the path when training your own public backbones for saving
    # path = "./trained_models/my_model.pth"
    # torch.save(model.state_dict(), path)

    # start testing
    print("---Start Evaluation---\n")
    model.eval()
    true_cost_list_groups = []
    pred_cost_list_groups = []
    if not args.training:
        model.load_state_dict(torch.load(args.model_path)) # load trained backbones

    mse_list = []
    model.eval()
    inf_carbon_pred, inf_carbon_true, mse_metric = get_carbon_from_PM(carbon_test_data)

    wass_dist_min, wass_dist_max = cal_wass_distance(sub_groups_demand_test)
    print("Wass distance of demands between groups: [", wass_dist_min, ", ", wass_dist_max, "]") # [ 0.029343624115525817 ,  0.5800256778880332 ]
    print("Lambda values for each group: ", lambda_list)

    print("MSE: ", mse_metric)
    for n in range(N):
        test_demands = sub_groups_demand_test[n]
        expanded_inf_carbon_pred = torch.full((len(test_demands), 1), inf_carbon_pred.item(),
                                              device=inf_carbon_pred.device,
                                              dtype=inf_carbon_pred.dtype)
        expanded_inf_carbon_true = torch.full((len(test_demands), 1), inf_carbon_true.item(),
                                              device=inf_carbon_true.device,
                                              dtype=inf_carbon_true.dtype).to('cuda')

        pred_action_list = calculate_action_in_cuda(test_demands, expanded_inf_carbon_pred, lambda_list[n])

        pred_cost = calculate_cost_in_cuda(test_demands, pred_action_list, expanded_inf_carbon_true, lambda_list[n])

        true_action_list = calculate_action_in_cuda(test_demands, expanded_inf_carbon_true, lambda_list[n])

        true_cost = calculate_cost_in_cuda(test_demands, true_action_list, expanded_inf_carbon_true, lambda_list[n])

        _, pred_cost_list = calculate_mean(pred_cost)
        _, true_cost_list = calculate_mean(true_cost)
        arr = pred_cost_list - true_cost_list

        pred_cost_list_groups.append(pred_cost_list)
        true_cost_list_groups.append(true_cost_list)


    pred_means = []
    diffs = []

    for n in range(0, len(pred_cost_list_groups)):
        pred_means.append(np.mean(pred_cost_list_groups[n]))
        diffs.append(np.mean(pred_cost_list_groups[n]) - np.mean(true_cost_list_groups[n]))

    print("Variance: ", np.var(diffs), "Means", np.mean(diffs))




