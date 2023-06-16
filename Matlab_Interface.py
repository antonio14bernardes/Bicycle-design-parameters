import subprocess
from xml.dom.minidom import parse
import pandas as pd

# Define the path to the xml file
model_file_path = r"C:\Users\anton\Desktop\TU Delft\Musculoskeletal Modeling and Simulation\Final Project\ME41006_BicycleModelFiles\ME41006_Bicycle_Model_MuscleDriven_Python.osim"
# Define the path to the xml file of the model with prescribed motion
prescribed_motion_model_file_path = r"C:\Users\anton\Desktop\TU Delft\Musculoskeletal Modeling and Simulation\Final Project\ME41006_BicycleModelFiles\ME41006_Bicycle_Model_MuscleDriven_Prescribed_Motion_Python.osim"
# Define the path to the model matlab script
model_script_path = r"C:\Users\anton\Desktop\TU Delft\Musculoskeletal Modeling and Simulation\Final Project\ME41006_BicycleModelFiles\ME41006_BicycleModel_Python.m"
# Define the path to the prescribed motion matlab script
prescribed_motion_model_script_path = r"C:\Users\anton\Desktop\TU Delft\Musculoskeletal Modeling and Simulation\Final Project\ME41006_BicycleModelFiles\ME41006_BicycleModel_Prescribed_Motion_Python.m"

# Define the path to the muscle-to-torques matlab script
muscle2torque_script_path = r"C:\Users\anton\Desktop\TU Delft\Musculoskeletal Modeling and Simulation\Final Project\ME41006_BicycleModelFiles\ReplaceMusclesByTorques_Python.m"

# Define the path to the model matlab script
moco_study_script_path = r"C:\Users\anton\Desktop\TU Delft\Musculoskeletal Modeling and Simulation\Final Project\ME41006_BicycleModelFiles\ME41006MocoStudy_Python.m"

# Define the path to the opensim Static Optimization setup path
opensim_SO_setup_path = r"C:\Users\anton\Desktop\TU Delft\Musculoskeletal Modeling and Simulation\Final Project\ME41006_BicycleModelFiles\Simulation Setups\Static Optimization Setup.xml"

# Define the path to the opensim Forward Dynamics setup path
opensim_FD_setup_path = r"C:\Users\anton\Desktop\TU Delft\Musculoskeletal Modeling and Simulation\Final Project\ME41006_BicycleModelFiles\Simulation Setups\Forward Dynamics Setup.xml"
opensim_prescribed_motion_FD_setup_path = r"C:\Users\anton\Desktop\TU Delft\Musculoskeletal Modeling and Simulation\Final Project\ME41006_BicycleModelFiles\Simulation Setups\Forward Dynamics Prescribed Motion Setup.xml"

def modify_model_parameters(new_crank_length, new_pelvis_height):
    # For normal model

    # Read the script.
    with open(model_script_path, 'r') as file:
        script = file.read()

    # Find the line with the crank_length_for_sim variable and change its value
    start_crank = script.find('crank_length_for_sim = ') + len('crank_length_for_sim = ')
    end_crank = script.find(';', start_crank)
    script = script[:start_crank] + str(new_crank_length) + script[end_crank:]

    # Find the line with the pelvis_height_for_sim variable and change its value
    start_pelvis = script.find('pelvis_height_for_sim = ') + len('pelvis_height_for_sim = ')
    end_pelvis = script.find(';', start_pelvis)
    script = script[:start_pelvis] + str(new_pelvis_height) + script[end_pelvis:]

    # Write the modified script back to the file.
    with open(model_script_path, 'w') as file:
        file.write(script)



    # For model with prescribed motion

    # Read the script.
    with open(prescribed_motion_model_script_path, 'r') as file:
        script = file.read()

    # Find the line with the crank_length_for_sim variable and change its value
    start_crank = script.find('crank_length_for_sim = ') + len('crank_length_for_sim = ')
    end_crank = script.find(';', start_crank)
    script = script[:start_crank] + str(new_crank_length) + script[end_crank:]

    # Find the line with the pelvis_height_for_sim variable and change its value
    start_pelvis = script.find('pelvis_height_for_sim = ') + len('pelvis_height_for_sim = ')
    end_pelvis = script.find(';', start_pelvis)
    script = script[:start_pelvis] + str(new_pelvis_height) + script[end_pelvis:]

    # Write the modified script back to the file.
    with open(prescribed_motion_model_script_path, 'w') as file:
        file.write(script)


def add_prescribed_motion():
    # Parse the .osim XML file
    doc = parse(prescribed_motion_model_file_path)

    # Find the 'JointSet'
    jointSets = doc.getElementsByTagName('JointSet')
    for jointSet in jointSets:
        # Find the 'PinJoint' named 'BracketToFrame'
        pinJoints = jointSet.getElementsByTagName('PinJoint')
        for pinJoint in pinJoints:
            if pinJoint.getAttribute('name') == 'BracketToFrame':
                bracketToFrame = pinJoint
                break

        # Find the 'Coordinate' named 'Bracket'
        coordinates = bracketToFrame.getElementsByTagName('Coordinate')
        for coordinate in coordinates:
            if coordinate.getAttribute('name') == 'Bracket':
                bracket = coordinate
                break

        # Create the new 'prescribed_function' and 'prescribed' nodes
        pf = doc.createElement('prescribed_function')
        lf = doc.createElement('LinearFunction')
        coeff = doc.createElement('coefficients')
        coeff.appendChild(doc.createTextNode('6.28319 0'))
        lf.appendChild(coeff)
        pf.appendChild(lf)
        bracket.appendChild(pf)

        prescribed = doc.createElement('prescribed')
        prescribed.appendChild(doc.createTextNode('true'))
        bracket.appendChild(prescribed)

    # Write the modified .osim XML file back to disk
    with open(prescribed_motion_model_file_path, 'w') as f:
        f.write(doc.toprettyxml())




def extract_muscle_params_minidom(file_path):
    # Parse the XML file.
    dom_tree = parse(file_path)

    # Get all Millard2012EquilibriumMuscle elements.
    muscles = dom_tree.getElementsByTagName('Millard2012EquilibriumMuscle')

    muscle_params = {}
    for muscle in muscles:
        muscle_name = muscle.getAttribute('name')

        # Extract max_isometric_force and optimal_fiber_length.
        max_isometric_force = muscle.getElementsByTagName('max_isometric_force')[0].firstChild.nodeValue
        optimal_fiber_length = muscle.getElementsByTagName('optimal_fiber_length')[0].firstChild.nodeValue

        # Append the muscle parameters to the list.
        muscle_params[muscle_name]={
            'max_isometric_force': float(max_isometric_force),
            'optimal_fiber_length': float(optimal_fiber_length),
        }

    return muscle_params


def run_matlab_script(script_path, verbose = True):
    if verbose:
        subprocess.run(["matlab", "-batch", "run('{}')".format(script_path)])
    else:
        subprocess.run(["matlab", "-batch", "run('{}')".format(script_path)], stdout=subprocess.DEVNULL)

def run_opensim_tool(tool_setup_path, verbose = True):
    if verbose:
        subprocess.run(["opensim-cmd", "run-tool", tool_setup_path])
    else:
        subprocess.run(["opensim-cmd", "run-tool", tool_setup_path], stdout=subprocess.DEVNULL)

def get_modified_model(verbose = True):
    run_matlab_script(model_script_path, verbose)
    run_matlab_script(prescribed_motion_model_script_path, verbose)

def get_torques_model(verbose = True):
    run_matlab_script(muscle2torque_script_path, verbose)

def get_motion_data_with_moco(verbose = True):
    run_matlab_script(moco_study_script_path, verbose)

def get_motion_data(verbose = True):
    run_opensim_tool(opensim_prescribed_motion_FD_setup_path, verbose)

def get_controls(verbose = True):
    run_opensim_tool(opensim_SO_setup_path, verbose)

def get_FD_simulation(verbose = True):
    run_opensim_tool(opensim_FD_setup_path, verbose)

def xml_to_dataframe(file_name):
    # Parse XML file
    xmldoc = parse(file_name)

    # Get elements
    nodes = xmldoc.getElementsByTagName('ControlLinearNode')

    # Initialize lists for data
    t_values = []
    value_values = []

    # Loop through nodes
    for node in nodes:
        # Get 't' values
        t = node.getElementsByTagName('t')[0].childNodes[0].data
        t_values.append(float(t))

        # Get 'value' values
        value = node.getElementsByTagName('value')[0].childNodes[0].data
        value_values.append(float(value))

    # Create pandas DataFrame
    df = pd.DataFrame({
        'time': t_values,
        'value': value_values
    })

    return df

if __name__ == '__main__':

    new_crank_length = 0.25
    new_pelvis_height = 0.75
    modify_model_parameters(new_crank_length, new_pelvis_height)
    get_modified_model(verbose =True)
    print("Got the modified model")
    get_torques_model(verbose = True)
    print("Got the torques version of the modified model")
    get_motion_data(verbose = True)
    print("Got the motion from the Moco study")
    get_controls(verbose= True)
    print("Got the Static Optimization outputs")
    get_FD_simulation(verbose = True)
    print("Got the Forward Dynamics")


