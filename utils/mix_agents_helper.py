import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def read_ev_data():
    ev_data_path = "../data/caltech_ev_dataset_detail.csv"
    total_ev_data = pd.read_csv(ev_data_path)
    total_ev_data['start_time'] = pd.to_datetime(total_ev_data['start_time'])
    total_ev_data['done_charge'] = pd.to_datetime(total_ev_data['done_charge'])
    total_ev_data['duration'] = total_ev_data['done_charge'] - total_ev_data['start_time']
    candidate_ev_data = total_ev_data[total_ev_data['start_time'] < total_ev_data['done_charge']]
    candidate_ev_data['duration'] = pd.to_timedelta(candidate_ev_data['duration'])
    duration_seconds = candidate_ev_data['duration'].dt.total_seconds().astype(int).to_numpy()
    candidate_index = np.where((duration_seconds / 3600 >= 3) & (duration_seconds / 3600 <= 12))[0]

    return candidate_ev_data.iloc[candidate_index]

def read_iphone_data():
    charging_demands = [5.18, 4.25, 4.44, 5.3, 5.3, 5.45, 5.73, 5.92, 6.9, 6.91, 10.45, 6.21, 7.45, 11.1, 6.96, 10.28, 10.35, 11.16]
    charging_efficiency = 2
    charging_hrs = [round(x/charging_efficiency) for x in charging_demands]
    iphone_df = pd.DataFrame({'charging_demands': charging_demands, 'charging_hrs': charging_hrs})
    return iphone_df

def read_dc_data():
    demand_path = "../data/azure_total_demand.csv"
    df = pd.read_csv(demand_path)[1:]
    df_demand = df.reset_index(drop=True)
    df_demand['Time'] = pd.to_datetime(df_demand['Time'])

    # Group the DataFrame by year, month, day, hour, and minute
    grouped_df = df_demand.groupby([df_demand['Time'].dt.year, df_demand['Time'].dt.month, df_demand['Time'].dt.day,
                                    df_demand['Time'].dt.hour]).sum()

    scaler = MinMaxScaler(feature_range=(0, 10))
    df_scaled = pd.DataFrame(scaler.fit_transform(grouped_df))
    return df_scaled.iloc[:][0]

def expand_iphone_dataF(df, batch_size):
    expanded_df = pd.concat([df] * 7, ignore_index=True)
    remaining_rows = batch_size - len(expanded_df)
    expanded_df = pd.concat([expanded_df, df.head(remaining_rows)], ignore_index=True)

    return expanded_df

def calculate_mean(cost_list):
    array = np.array([tensor.cpu().numpy() for tensor in cost_list])
    normalized_array = (array - np.min(array)) / (np.max(array) - np.min(array))

    return normalized_array, array
