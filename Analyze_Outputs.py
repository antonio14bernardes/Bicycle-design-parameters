from Handle_files import read_sto_file
from Matlab_Interface import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

FD_output_data_path = r"C:\Users\anton\Desktop\TU Delft\Musculoskeletal Modeling and Simulation\Final Project\ME41006_BicycleModelFiles\Simulation Outputs\Forward Dynamics Prescribed Motion\ME41006_Bicycle_modelMuscleDriven_states.sto"
reaction_forces_output_data_path = r"C:\Users\anton\Desktop\TU Delft\Musculoskeletal Modeling and Simulation\Final Project\ME41006_BicycleModelFiles\Simulation Outputs\Analysis\ME41006_Bicycle_modelMuscleDriven_JointReaction_ReactionLoads.sto"
muscle_forces_output_data_path = r"C:\Users\anton\Desktop\TU Delft\Musculoskeletal Modeling and Simulation\Final Project\ME41006_BicycleModelFiles\Simulation Outputs\Analysis\ME41006_Bicycle_modelMuscleDriven_MuscleAnalysis_ActiveFiberForce.sto"
activation_output_data_path = r"C:\Users\anton\Desktop\TU Delft\Musculoskeletal Modeling and Simulation\Final Project\ME41006_BicycleModelFiles\Simulation Outputs\Static Optimization\ME41006_Bicycle_modelMuscleDriven_StaticOptimization_activation.sto"
q_output_data_path = r"C:\Users\anton\Desktop\TU Delft\Musculoskeletal Modeling and Simulation\Final Project\ME41006_BicycleModelFiles\Simulation Outputs\Analysis\ME41006_Bicycle_modelMuscleDriven_Kinematics_q.sto"
u_output_data_path = r"C:\Users\anton\Desktop\TU Delft\Musculoskeletal Modeling and Simulation\Final Project\ME41006_BicycleModelFiles\Simulation Outputs\Analysis\ME41006_Bicycle_modelMuscleDriven_Kinematics_u.sto"
fiber_velocity_output_data_path = r"C:\Users\anton\Desktop\TU Delft\Musculoskeletal Modeling and Simulation\Final Project\ME41006_BicycleModelFiles\Simulation Outputs\Analysis\ME41006_Bicycle_modelMuscleDriven_MuscleAnalysis_FiberVelocity.sto"
normalized_fiber_velocity_output_data_path = r"C:\Users\anton\Desktop\TU Delft\Musculoskeletal Modeling and Simulation\Final Project\ME41006_BicycleModelFiles\Simulation Outputs\Analysis\ME41006_Bicycle_modelMuscleDriven_MuscleAnalysis_NormFiberVelocity.sto"
fiber_length_output_data_path = r"C:\Users\anton\Desktop\TU Delft\Musculoskeletal Modeling and Simulation\Final Project\ME41006_BicycleModelFiles\Simulation Outputs\Analysis\ME41006_Bicycle_modelMuscleDriven_MuscleAnalysis_FiberLength.sto"
states_output_data_path = r"C:\Users\anton\Desktop\TU Delft\Musculoskeletal Modeling and Simulation\Final Project\ME41006_BicycleModelFiles\Simulation Outputs\Analysis\ME41006_Bicycle_modelMuscleDriven_StatesReporter_states.sto"

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

    return 0 if index == 0 else index - 1

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

            interp = (next_value - prev_value) / 2

            arrays[muscle_data_i, time_i] = interp + prev_value

    new_df['time'] = time_array

    for i, muscle_data in enumerate(total_muscle_data):
        new_df[muscle_data] = arrays[i]

    return new_df

def handle_missmatched_times_metabolic(keep_time, change_time):
    new_df = pd.DataFrame()
    time_array = keep_time['time']
    arrays = np.empty((len(total_muscles), len(time_array)))

    for time_i, ti in enumerate(time_array):
        index_prev = get_closest_index_floored(ti, change_time['time'])
        index_next = index_prev + 1


        for muscle_data_i, muscle_data in enumerate(total_muscles):
            prev_value = change_time[muscle_data][index_prev]
            next_value = change_time[muscle_data][index_next]
            delta_t = change_time['time'][index_next] - change_time['time'][index_prev]

            interp = (next_value - prev_value) / 2

            arrays[muscle_data_i, time_i] = interp + prev_value

    new_df['time'] = time_array

    for i, muscle_data in enumerate(total_muscles):
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


def compute_contraction_velocities(fiber_length_df, total_muscles):
    contraction_vel_df = pd.DataFrame()
    for key in total_muscles:
        contraction_velocity = - differentiate(fiber_length_df[key], fiber_length_df['time'])
        contraction_vel_df[key] = np.copy(contraction_velocity)

    return contraction_vel_df

def get_activation(states_df):
    activation_df = pd.DataFrame()
    activation_df['time'] = states_df['time']
    for muscle in total_muscles:
        key = complete_muscle_data_name(muscle) + "/activation"
        activation_df[muscle] = states_df[key]

    return activation_df


def compute_individual_metabolic_energy_cost(fiber_length_df, muscle_force_df, activation_df, muscle_params_dict):
    keys = muscle_force_df.keys()
    metabolic_energy_df = pd.DataFrame()
    reduced_fiber_length_df = handle_missmatched_times_metabolic(activation_df,fiber_length_df)
    reduced_muscle_force_df = handle_missmatched_times_metabolic(activation_df,muscle_force_df)
    metabolic_energy_df['time'] = activation_df['time']
    contraction_velocities_df = compute_contraction_velocities(reduced_fiber_length_df, total_muscles)


    #

    for key in total_muscles:
        force = reduced_muscle_force_df[key]
        activation = activation_df[key]
        fiber_velocity = contraction_velocities_df[key]

        params_dict = muscle_params_dict[key]
        max_isometric_force = params_dict['max_isometric_force']
        optimal_fiber_length = params_dict['optimal_fiber_length']

        norm_contraction_velocity = fiber_velocity/optimal_fiber_length
        norm_vel_only_contraction = np.clip(norm_contraction_velocity, a_min = 0, a_max = None)
        norm_force = force / max_isometric_force

        normalization = 0.05
        metabolic_energy = activation * norm_force * norm_vel_only_contraction / normalization

        metabolic_energy_df[key] = np.copy(metabolic_energy)

    return metabolic_energy_df

def compute_total_metabolic_cost(metabolic_cost_data):
    total_per_muscle = metabolic_cost_data.drop(labels = "time", axis = 1).sum(axis = 0).to_numpy()
    total = np.sum(total_per_muscle)

    return total

def compute_power_output(u_df, q_df, reaction_df, crank_length):

    bracket_degrees = q_df['Bracket']
    bracket_radians = np.deg2rad(bracket_degrees)
    bracket_position = np.array([[np.cos(curr_angle) * crank_length, np.sin(curr_angle) * crank_length, 0] for curr_angle in bracket_radians])
    bracket_rotation_vel = np.array([[0,0, np.deg2rad(vel)] for vel in u_df['Bracket'].to_numpy()])

    forces_x_r = -reaction_df["mtp_r_on_toes_r_in_ground_fx"]
    forces_y_r = -reaction_df["mtp_r_on_toes_r_in_ground_fy"]
    forces_x_l = -reaction_df["mtp_l_on_toes_l_in_ground_fx"]
    forces_y_l = -reaction_df["mtp_l_on_toes_l_in_ground_fy"]

    force_vectors_r = np.array([[f_x, f_y, 0] for f_x, f_y in zip(forces_x_r, forces_y_r)])
    force_vectors_l = np.array([[f_x, f_y, 0] for f_x, f_y in zip(forces_x_l, forces_y_l)])


    torques = np.cross(-bracket_position, force_vectors_r) + np.cross(bracket_position, force_vectors_l)

    power = np.array([np.dot(torque, bracket_rot_vel) for torque, bracket_rot_vel in zip(torques, bracket_rotation_vel)])

    power_df = pd.DataFrame({'time': u_df['time'], 'power': power})
    return power_df

def get_work(u_df, q_df, reaction_df, crank_length):
    power_df = compute_power_output(u_df, q_df, reaction_df, crank_length)

    times = power_df['time'].to_numpy()
    deltas = times[1:] - times[:-1]
    powers = power_df['power'].to_numpy()[:-1]

    energy = 0

    for delta, power in zip(deltas, powers):
        energy += delta * power

    return energy




"""-------------------- Data Processing --------------------"""

def check_validity_IQR(value, Q1, Q3, threshold):
    IQR = Q3 - Q1

    limit_sup = Q3 + threshold * IQR
    limit_inf = Q1 - threshold * IQR

    return value>limit_inf and value<limit_sup

def remove_outliers(values, index_to_analyze, threshold):

    Q1 = np.percentile(values[index_to_analyze], 25, interpolation = 'midpoint')
    Q3 = np.percentile(values[index_to_analyze], 75, interpolation = 'midpoint')
    print(Q1,Q3)
    indexes_to_keep = [i for i in range(len(values[index_to_analyze])) if check_validity_IQR(values[index_to_analyze][i], Q1, Q3, threshold)]

    new_values = (values.T[indexes_to_keep, :]).T


    return new_values

Q1_from_experiments = 11.212081036250716
Q3_from_experiments = 19.153469553983143
threshold_from_experiments = 1.2


if __name__ == "__main__":
    """ Loading data """
    optimal_fitness = np.load("Sprint_fitness.npy")[-1]
    optimal_crank_length, optimal_seat_height = np.load("Sprint_position.npy")

    iteration_values_crank = np.load('Sprint_iterated_values_crank.npy')
    metabolic_energy_crank = np.load('Sprint_metabolic_energy_crank.npy')

    iteration_values_seat = np.load('Sprint_iterated_values_seat.npy')
    metabolic_energy_seat = np.load('Sprint_metabolic_energy_seat.npy')

    """ Normalizing values """
    reltive_iteration_values_crank = iteration_values_crank/optimal_crank_length
    relative_metabolic_energy_crank = metabolic_energy_crank/optimal_fitness

    reltive_iteration_values_seat = iteration_values_seat/optimal_seat_height
    relative_metabolic_energy_seat = metabolic_energy_seat/optimal_fitness

    """ Removing outliers """
    crank_values= np.array([reltive_iteration_values_crank, relative_metabolic_energy_crank])
    seat_values = np.array([reltive_iteration_values_seat, relative_metabolic_energy_seat])

    clean_it_crank, clean_met_crank = remove_outliers(crank_values, 1, 1.2)
    clean_it_seat, clean_met_seat = remove_outliers(seat_values, 1, 1.2)


    """ Filtering """
    window_size = 17
    poly_order = 3

    smooth_met_crank = savgol_filter(clean_met_crank, window_size, poly_order)
    smooth_met_seat = savgol_filter(clean_met_seat, window_size, poly_order)

    #to_plot = "raw"
    #to_plot = "processed"
    to_plot = "fitness"

    if to_plot == "raw":
        plt.plot(reltive_iteration_values_crank,relative_metabolic_energy_crank, label = "crank length")
        plt.plot(reltive_iteration_values_seat,relative_metabolic_energy_seat, label = "seat height")

        plt.title("Raw SMEC sensitivity in sprint task")
        plt.ylabel("Normalized raw SMEC")
        plt.xlabel("Normalized dimension (crank length or seat height)")

        plt.legend()

    elif to_plot == "processed":
        plt.plot(clean_it_crank, smooth_met_crank, label = "crank length")
        plt.plot(clean_it_seat, smooth_met_seat, label = "seat height")

        plt.title("SMEC sensitivity in sprint task")
        plt.ylabel("Normalized SMEC")
        plt.xlabel("Normalized dimension (crank length or seat height)")

        plt.legend()

    elif to_plot == "fitness":
        sprint_fitness = np.load("Sprint_fitness.npy")/np.load("Sprint_fitness.npy")[-1]
        long_distance_fitness = np.load("Long_distance_fitness.npy")/np.load("Long_distance_fitness.npy")[-1]

        plt.plot(sprint_fitness, label = "sprint task")
        plt.plot(long_distance_fitness, label = "long-distance task")

        plt.title("Optimization with PSO algorithm")
        plt.ylabel("Normalized fitness value (normalized SMEC)")
        plt.xlabel("Iteration")

        plt.legend()

    plt.show()