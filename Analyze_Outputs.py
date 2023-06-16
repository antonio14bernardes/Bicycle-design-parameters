from Handle_files import read_sto_file
from Matlab_Interface import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FD_output_data_path = r"C:\Users\anton\Desktop\TU Delft\Musculoskeletal Modeling and Simulation\Final Project\ME41006_BicycleModelFiles\Simulation Outputs\Forward Dynamics\ME41006_Bicycle_modelMuscleDriven_states.sto"
forces_output_data_path = r"C:\Users\anton\Desktop\TU Delft\Musculoskeletal Modeling and Simulation\Final Project\ME41006_BicycleModelFiles\Simulation Outputs\Static Optimization\ME41006_Bicycle_modelMuscleDriven_StaticOptimization_force.sto"
muscles = ["hamstrings", "bifemsh", "glut_max", "iliopsoas", "rect_fem", "vasti", "gastroc", "soleus", "tib_ant"]
legs = ["l", "r"]
total_muscles = [muscle + "_" + leg for muscle in muscles for leg in legs]
overall_data_names = ['fiber_length', 'activation']
total_muscle_data = [muscle + r"/" + data for muscle in total_muscles for data in overall_data_names]

def complete_muscle_data_name(muscle_data_name):
    return r'/forceset' + r'/' + muscle_data_name

def differentiate(y,x):
    dx = np.diff(x)
    dy = np.diff(y)
    dy_dx = dy/dx

    # Fill up last value as the same as the previous one
    dy_dx = np.append(dy_dx, np.array([dy_dx[-1]]))

    return dy_dx

def get_closest_index_floored(value, array): #For time
    index = -1
    larger = False
    while not larger:
        index += 1
        larger = True if (array[index] > value) or (index == len(array)-1) else False

    return index - 1

def handle_missmatched_times(keep_time, change_time):
    new_df = pd.DataFrame()
    time_array = keep_time['time']
    arrays = np.empty((len(total_muscle_data), len(time_array)))

    for time_i, ti in enumerate(time_array):
        index_prev = get_closest_index_floored(ti, change_time['time'])
        index_next = index_prev + 1


        for muscle_data_i, muscle_data in enumerate(total_muscle_data):
            prev_value = change_time[complete_muscle_data_name(muscle_data)][index_prev]
            next_value = change_time[complete_muscle_data_name(muscle_data)][index_next]
            delta_t = change_time['time'][index_next] - change_time['time'][index_prev]

            interp = (next_value - prev_value) / delta_t

            arrays[muscle_data_i, time_i] = interp

    new_df['time'] = time_array

    for i, muscle_data in enumerate(total_muscle_data):
        new_df[muscle_data] = arrays[i]

    return new_df

def remove_annoying_name_part(df, keys):  # USELESS WASTE IF TIME
    new_df = pd.DataFrame()

    for key in keys:
        if key == 'time':
            new_df['time'] = df['time']
        else:
            new_df[key] = df[complete_muscle_data_name(key)]

    return new_df


def compute_contraction_velocities(overall_df, total_muscles):
    new_keys = [muscle + r'/' + 'contraction_velocity' for muscle in total_muscles]
    length_keys = [muscle + r'/' + 'fiber_length' for muscle in total_muscles]

    for vel_key, length_key in zip(new_keys, length_keys):
        contraction_velocity = - differentiate(overall_df[length_key], overall_df['time'])
        overall_df[vel_key] = np.copy(contraction_velocity)

    return overall_df




def compute_individual_metabolic_energy_cost(overall_df, force_df, muscle_params_dict):
    keys = force_df.keys()
    metabolic_energy_df = pd.DataFrame()

    for key in keys:
        if key == 'time':
            metabolic_energy_df['time'] = force_df['time']
        else:
            force = force_df[key]
            activation = overall_df[key + r'/' + 'activation']
            velocity = overall_df[key + r'/' + 'contraction_velocity']

            params_dict = muscle_params_dict[key]
            max_isometric_force = params_dict['max_isometric_force']
            optimal_fiber_length = params_dict['optimal_fiber_length']


            vel_only_contraction = np.clip(velocity, a_min = 0, a_max = None)

            normalization = 100

            metabolic_energy = activation * (force/max_isometric_force) * (vel_only_contraction/optimal_fiber_length) / normalization

            metabolic_energy_df[key] = np.copy(metabolic_energy)

    return metabolic_energy_df

def compute_total_metabolic_cost(metabolic_cost_data):
    total_per_muscle = metabolic_cost_data.drop(labels = "time", axis = 1).sum(axis = 0).to_numpy()
    total = np.sum(total_per_muscle)

    return total

def compute_power_output(overall_df):
    wheel_inertia = 3 # Pay attention to changes to the wheel in the model
    bracket_inertia = 1 # Pay attention to changes to the bracket in the model

    wheel_rotation_vel = overall_df['/jointset/RearWheelToFrame/RearWheel/speed'].to_numpy()
    bracket_rotation_vel = overall_df['/jointset/BracketToFrame/Bracket/speed'].to_numpy()

    energy_final = (1/2) * wheel_inertia * wheel_rotation_vel[-1] ** 2 + \
                   (1/2) * bracket_inertia * bracket_rotation_vel[-1] ** 2

    return energy_final

if __name__ == "__main__":
    overall_output_data = read_sto_file(FD_output_data_path)
    force_output_data = read_sto_file(forces_output_data_path)
    muscle_params_dict = extract_muscle_params_minidom(model_file_path)

    muscles_output_data = handle_missmatched_times(force_output_data, overall_output_data)
    muscles_output_data = compute_contraction_velocities(muscles_output_data, total_muscles)
    metabolic_energy_individual = compute_individual_metabolic_energy_cost(muscles_output_data, force_output_data, muscle_params_dict)
    metabolic_energy = compute_total_metabolic_cost(metabolic_energy_individual)

    energy_output = compute_power_output(overall_output_data)

    print(metabolic_energy, energy_output)