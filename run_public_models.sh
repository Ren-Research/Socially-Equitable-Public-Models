# Data Center Workload

## Same Lambda, Different Groups
python main_dc_workload.py --diff_group_dist --model_path 'trained_public_models/dc/s_lambda_d_dist/equitable_q1.pth'
python main_dc_workload.py --diff_group_dist --model_path 'trained_public_models/dc/s_lambda_d_dist/equitable_q3.pth'
python main_dc_workload.py --diff_group_dist --model_path 'trained_public_models/dc/s_lambda_d_dist/equitable_q20.pth'
python main_dc_workload.py --diff_group_dist --model_path 'trained_public_models/dc/s_lambda_d_dist/mse_baseline.pth'

## Different Lambda, Similar Groups
python main_dc_workload.py --diff_lambda --model_path 'trained_public_models/dc/d_lambda_s_dist/equitable_q1.pth'
python main_dc_workload.py --diff_lambda --model_path 'trained_public_models/dc/d_lambda_s_dist/equitable_q1.1.pth'
python main_dc_workload.py --diff_lambda --model_path 'trained_public_models/dc/d_lambda_s_dist/equitable_q1.5.pth'
python main_dc_workload.py --diff_lambda --model_path 'trained_public_models/dc/d_lambda_s_dist/mse_baseline.pth'

## Different Lambda, Different Groups
python main_dc_workload.py --diff_lambda --diff_group_dist --model_path 'trained_public_models/dc/d_lambda_d_dist/equitable_q1.pth'
python main_dc_workload.py --diff_lambda --diff_group_dist --model_path 'trained_public_models/dc/d_lambda_d_dist/equitable_q3.pth'
python main_dc_workload.py --diff_lambda --diff_group_dist --model_path 'trained_public_models/dc/d_lambda_d_dist/equitable_q10.pth'
python main_dc_workload.py --diff_lambda --diff_group_dist --model_path 'trained_public_models/dc/d_lambda_d_dist/mse_baseline.pth'


# EV Charging

## Different Groups
python main_ev_charging.py --model_path 'trained_public_models/ev/d_dist/equitable_q20.pth' --diff_group_dist
python main_ev_charging.py --model_path 'trained_public_models/ev/d_dist/equitable_q30.pth' --diff_group_dist
python main_ev_charging.py --model_path 'trained_public_models/ev/d_dist/equitable_q40.pth' --diff_group_dist
python main_ev_charging.py --model_path 'trained_public_models/ev/d_dist/mse_baseline.pth' --diff_group_dist

## Similar Groups
python main_ev_charging.py --model_path 'trained_public_models/ev/s_dist/equitable_q20.pth'
python main_ev_charging.py --model_path 'trained_public_models/ev/s_dist/equitable_q30.pth'
python main_ev_charging.py --model_path 'trained_public_models/ev/s_dist/equitable_q40.pth'
python main_ev_charging.py --model_path 'trained_public_models/ev/s_dist/mse_baseline.pth'
