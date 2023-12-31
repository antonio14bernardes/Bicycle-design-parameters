import numpy as np
from PSO import system, get_performance_metrics
import matplotlib.pyplot as plt
from Analyze_Outputs import remove_outliers
from scipy.signal import savgol_filter

""" Define the +- percentage interval around optimal values where to analyse sensitivity """
crank_length_variation = 5 # % of the optimal value to change + and -
seat_height_variation = 5 # % of the optimal value to change + and -

""" Write here the optimal values for each dimension """
"""task = 'Power'
if task == 'Power':
    optimal_crank_length, optimal_seat_height = np.load('Best_position_Power.npy')
elif task == 'Long_Distance':
    optimal_crank_length, optimal_seat_height = np.load('Best_position_Long_Distance.npy')"""


# TEMPORARY!!!!!!!!!!!!!!!!!!!!!!!!!
#optimal_crank_length = 0.18305553
#optimal_seat_height = 0.80323285

optimal_crank_length = 0.16644901
optimal_seat_height = 0.70142167

""" Define the resolution to study as a percentage of optimal values """
crank_length_resolution = 0.5 # %
seat_height_resolution = 0.5 # %

""" Get the range of exploration """
crank_length_range = [optimal_crank_length * (1 - crank_length_variation/100), optimal_crank_length * (1 + crank_length_variation/100)]
seat_height_range = [optimal_seat_height * (1 - seat_height_variation/100), optimal_seat_height * (1 + seat_height_variation/100)]
#print("crank_length_range", crank_length_range)
#print("seat_height_range",seat_height_range)

""" Get the specific values for exploration """
crank_length_step = optimal_crank_length * crank_length_resolution / 100
seat_height_step = optimal_seat_height * seat_height_resolution / 100
#print("crank_length_step", crank_length_step)
#print("seat_height_step",seat_height_step)

crank_length_values = np.arange(start = crank_length_range[0], stop = crank_length_range[1], step = crank_length_step)
seat_height_values = np.arange(start = seat_height_range[0], stop = seat_height_range[1], step = seat_height_step)
#print("crank_length_values", crank_length_values)
#print("seat_height_values", seat_height_values)

""" Define function for a single variable study """
def run_single_sensitivity_analysis(index_to_iterate, fixed_value, iteration_values_array):
    ks = [0, 0] # Initialize with random values

    metabolic_energy = []
    final_energy = []

    fixed_index = abs(index_to_iterate - 1)
    ks[fixed_index] = fixed_value

    num_values_to_iterate = len(iteration_values_array)

    for i, iterated_value in enumerate(iteration_values_array):
        print("\nStarting iteration %s out of a total of %s\n" % (i + 1, num_values_to_iterate))
        ks[index_to_iterate] = iterated_value


        new_metabolic_energy, new_energy_final = get_performance_metrics(ks, system)


        metabolic_energy.append(new_metabolic_energy)
        final_energy.append(new_energy_final)

    np.save('iterated_values.npy', iteration_values_array)
    np.save('metabolic_energy.npy', metabolic_energy)
    np.save('final_energy.npy', final_energy)

    print("Saved arrays")

    plt.plot(iteration_values_array,metabolic_energy)
    plt.plot(iteration_values_array,final_energy)
    plt.show()



def main():
    #run_single_sensitivity_analysis(0, optimal_seat_height, crank_length_values)

    iteration_values_array = np.load('Long_distance_iterated_values_crank.npy')
    metabolic_energy = np.load('Long_distance_metabolic_energy_crank.npy')
    #final_energy = np.load('final_energy.npy')

    values = np.array([iteration_values_array, metabolic_energy])
    filtered_x, filtered_y = remove_outliers(values, 1, 1.2)

    window_size = 15
    poly_order = 3

    # Apply filter
    smooth_y = savgol_filter(filtered_y, window_size, poly_order)
    plt.plot(filtered_x,filtered_y)
    plt.plot(filtered_x, smooth_y)
    plt.show()

if __name__ == "__main__":
    main()