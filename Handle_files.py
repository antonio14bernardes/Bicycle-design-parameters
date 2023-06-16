import pandas as pd
import matplotlib.pyplot as plt

def read_sto_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        end_header = 0
        for i, line in enumerate(lines):
            if 'endheader' in line:
                end_header = i
                break

    df = pd.read_csv(file_path, sep='\t', skiprows=end_header + 1)

    df.columns = df.columns.str.strip()

    return df

def write_sto_file(df, file_path):
    with open(file_path, 'w') as f:
        f.write('Storage Version=1\n')
        f.write('nRows=%s\n' % df.shape[0])
        f.write('nColumns=%s\n' % df.shape[1])
        f.write('inDegrees=yes\n')
        f.write('endheader\n')

    df.to_csv(file_path, sep='\t', mode='a', index=False)

def global_path(global_part, file_name, extension):
    return global_part + "\\" + file_name + extension

lab2_path = r"C:\Users\anton\Desktop\TU Delft\Musculoskeletal Modeling and Simulation\Lab 2"
sto_extension = ".sto"
mot_extension = ".mot"
default_file = "Tug_of_War_TemplateMuscle_controls"



""" For Question C """

def run_Lab2_QC():
    # Setting up paths
    outputs_QC_path = r"C:\Users\anton\Desktop\TU Delft\Musculoskeletal Modeling and Simulation\Lab 2\Outputs"
    outputs_list = ["Outputs ratio 0.5", "Outputs ratio 1", "Outputs ratio 2", "Outputs ratio 4", "Outputs ratio 7"]
    opensim_path = "/forceset/LeftMuscle|"
    output_name = "tendon_force"

    # Experimental data
    ratios = [0.5, 1, 2, 4, 7]

    # Setting up DataFrames
    dfs_list = []
    for ratio_output in outputs_list:
        file_path = global_path(outputs_QC_path, ratio_output, sto_extension)
        new_df = read_sto_file(file_path)
        new_df.rename(columns = {opensim_path + output_name: output_name}, inplace = True)
        dfs_list.append(new_df)

    def plots():
        # Setup plot
        line_colors = ['#32CD32','#1E90FF', '#9400D3', '#DC143C', '#FFD700']
        line_styles = [((4, (5, 3))), ((0, (5, 3))), ((0, (5, 3))), ((0, (5, 3))), ((0, (5, 3)))]
        xlim = 0.5

        plt.figure(dpi=120)
        for df, color, style, ratio in zip(dfs_list, line_colors, line_styles, ratios):
            plt.plot(df["time"], df[output_name], linestyle=style, color = color, label='tf ratio = ' + str(ratio))

        plt.xlim(0,xlim)
        plt.ylabel('tendon force (N)')
        plt.xlabel('time (s)')
        plt.title('Tendon force over time for different tendon-to-fiber (tf) ratios')
        plt.legend()

        plt.show()