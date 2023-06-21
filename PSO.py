import numpy as np
import matplotlib.pyplot as plt
import keyboard
from Matlab_Interface import *
from Analyze_Outputs import *





def main():
    #INPUT VALUES

    #Optimization
    num_its=15      #number of iterations

    #Particles
    num_particles=8   #number of particles
    w=[0.9,0.4]         #inertia
    c1=[2.5,0.5]        #memory
    c2=[0.5,2.5]        #cooperation

    w_update=linear_update
    c1_update=linear_update
    c2_update=linear_update

    #Limits
    pos_upper_limit=[0.3,0.9]
    pos_lower_limit=[0.1, 0.6]
    vel_upper_limit=[0.05, 0.05]
    vel_lower_limit=[- vel for vel in vel_upper_limit]

    #Fitness
    fitness_fnc=fitness
    power_output_penalty=2 #Best for trace 10
    metabolic_energy_cost_penalty=1#Best for trace 10000

    fitness_penalties=[power_output_penalty, metabolic_energy_cost_penalty]

    #System
    system_fnc=system

    #Initial Velocity
    initially_static=True

    best_particle, best_global_fitness =optimize(num_particles,num_its,w,c1,c2,pos_upper_limit,pos_lower_limit,vel_upper_limit,
                           vel_lower_limit,fitness_fnc,fitness_penalties,system_fnc,w_update,c1_update,c2_update,
                           initially_static)


    print('Best Position',best_particle.best_position[-1])
    print('Best Fitnesses',best_global_fitness)

    print("Running system with the best position")
    system_fnc(best_particle.best_position[-1])
    print("All files have been updated")

    plt.plot(best_global_fitness)
    plt.show()

    np.save('Best_fitness.npy',np.array(best_global_fitness))
    np.save('Best_position.npy',np.array(best_particle.best_position[-1]))

    print("Saved arrays")


class Particle:
    def __init__(self,num_its,w,c1,c2,pos_upper_limit,pos_lower_limit,vel_upper_limit,vel_lower_limit,fitness_fnc,
                 fitness_penalties,system_fnc,w_update_fnc,c1_udpate_fnc,c2_update_fnc,initially_static=False):
        self.fitness_fnc=fitness_fnc
        self.fitness_penalties=fitness_penalties
        self.system_fnc=system_fnc
        self.w_init,self.w_final=w
        self.c1_init,self.c1_final=c1
        self.c2_init,self.c2_final=c2
        self.w_update_fnc=w_update_fnc
        self.c1_update_fnc=c1_udpate_fnc
        self.c2_update_fnc=c2_update_fnc

        self.pos_upper_limit=pos_upper_limit
        self.pos_lower_limit=pos_lower_limit
        self.vel_upper_limit=vel_upper_limit
        self.vel_lower_limit=vel_lower_limit

        self.position=np.array(
            [[np.random.uniform(low=low,high=high) for low,high in zip(pos_lower_limit,pos_upper_limit)]])
        if initially_static:
            self.velocity=np.zeros_like(self.position)
        else:
            self.velocity=np.array(
                [[np.random.uniform(low=low,high=high) for low,high in zip(vel_lower_limit,vel_upper_limit)]])
        init_fit=fitness_fnc(self.position[0],fitness_penalties,system_fnc)
        self.fitness=[init_fit]
        self.best_fitness=[init_fit]
        self.best_position=[self.position[0]]
        self.iteration=0
        self.num_its=num_its

    def update(self,latest_best_global_pos):
        self.w=self.w_update_fnc(self.num_its,self.iteration,self.w_init,self.w_final)
        self.c1=self.c1_update_fnc(self.num_its,self.iteration,self.c1_init,self.c1_final)
        self.c2=self.c2_update_fnc(self.num_its,self.iteration,self.c2_init,self.c2_final)

        new_velocity=self.w*self.velocity[-1]+self.c1*np.random.uniform(0,1)*(self.best_position[-1]-self.position[-1])\
            +self.c2*np.random.uniform(0,1)*(latest_best_global_pos-self.position[-1])
        new_velocity=clip(new_velocity,self.vel_upper_limit,self.vel_lower_limit)
        new_position=self.position[-1]+new_velocity
        new_position=clip(new_position,self.pos_upper_limit,self.pos_lower_limit)
        self.velocity=np.append(self.velocity,[new_velocity],axis=0)
        self.position=np.append(self.position,[new_position],axis=0)

        new_fit=self.fitness_fnc(self.position[-1],self.fitness_penalties,self.system_fnc)
        self.fitness.append(new_fit)
        if self.fitness[-1]<self.best_fitness[-1] and check_validity_IQR(self.fitness[-1], Q1_from_experiments, Q3_from_experiments, threshold_from_experiments):
            self.best_fitness.append(self.fitness[-1])
            self.best_position.append(self.position[-1])
        self.iteration+=1


def system(ks):
    task = 'long-distance'

    new_crank_length, new_pelvis_height = ks
    rot_vel, resistance = get_velocity_and_resistance(task)
    modify_model_parameters(new_crank_length, new_pelvis_height)
    get_modified_model(verbose = False)
    add_prescribed_motion(rot_vel)
    add_prescribed_force(model_file_path, [resistance, resistance, resistance])
    add_prescribed_force(prescribed_motion_model_file_path, [resistance, resistance, resistance])
    add_constraint_element(model_file_path, 'ankle_r')
    add_constraint_element(model_file_path, 'ankle_l')
    print("Got the modified model")
    get_motion_data(verbose = False)
    print("Got the motion for current model")
    get_controls(verbose= False)
    print("Got the Static Optimization outputs")
    get_FD_simulation(verbose = False)
    print("Got the Forward Dynamics")

def get_performance_metrics(ks, syst_fnc):
    syst_fnc(ks)

    muscle_force_output_data = read_sto_file(muscle_forces_output_data_path)
    activation_output_data = read_sto_file(activation_output_data_path)
    fiber_length_output_data = read_sto_file(fiber_length_output_data_path)
    muscle_params_dict = extract_muscle_params_minidom(model_file_path)
    individual_metabolic_energy_cost_df = compute_individual_metabolic_energy_cost(fiber_length_output_data, muscle_force_output_data, activation_output_data, muscle_params_dict)
    metabolic_energy_cost = compute_total_metabolic_cost(individual_metabolic_energy_cost_df)

    reaction_forces_output_data = read_sto_file(reaction_forces_output_data_path)
    u_output_data = read_sto_file(u_output_data_path)
    q_output_data = read_sto_file(q_output_data_path)

    energy_output = get_work(u_output_data, q_output_data, reaction_forces_output_data, 0.15)
    metabolic_energy_cost = compute_total_metabolic_cost(individual_metabolic_energy_cost_df)

    return metabolic_energy_cost, energy_output

def fitness(ks, penalties, syst_fnc):

    metabolic_energy, energy_output = get_performance_metrics(ks, syst_fnc)
    # Compute fitness
    power_output_penalty, metabolic_energy_cost_penalty = penalties
    fitness = metabolic_energy

    return fitness

def linear_update(num_its,its,initial,final):
    return (final-initial)*its/(num_its-1)+initial

def inertia_update(num_its,its,initial,final):
    return (initial-final)-((num_its-its)/num_its)*final

def clip(arr,max,min):
    return np.clip(arr,min,max)

def optimize(num_particles,num_its,w,c1,c2,pos_upper_limit,pos_lower_limit,vel_upper_limit,vel_lower_limit,fitness_fnc,
             fitness_penalties,system_fnc,w_update_fnc,c1_udpate_fnc,c2_update_fnc,initially_static=False):
    key_press=False
    particles=[]
    print('Initializing Particles')
    for i in range(num_particles):
        print('Creating particle',i,'out of',num_particles)
        particles.append(Particle(num_its,w,c1,c2,pos_upper_limit,pos_lower_limit,vel_upper_limit,vel_lower_limit,
                                  fitness_fnc,fitness_penalties,system_fnc,w_update_fnc,c1_udpate_fnc,c2_update_fnc,initially_static))
    best_global_fitness=[particles[0].best_fitness[-1]]
    best_global_position=[particles[0].best_position[-1]]
    best_particle=particles[0]
    for i in range(1,len(particles)):
        if particles[i].best_fitness[-1]<best_global_fitness[-1] and check_validity_IQR(particles[i].best_fitness[-1], Q1_from_experiments, Q3_from_experiments, threshold_from_experiments):
            best_particle=particles[i]
            best_global_position.append(particles[i].position[-1])
            best_global_fitness.append(particles[i].best_fitness[-1])
        else:
            best_global_fitness.append(best_global_fitness[-1])

    print('Initialization Completed. Proceding to Optimization')

    for it in range(num_its):
        print('Iteration', it, 'out of', num_its)
        for i in range(len(particles)):
            particles[i].update(best_global_position[-1])
            if keyboard.is_pressed('ctrl+3'):
                key_press=True
                print('Keyboard Interrupt (alt key pressed)')
                break
        if key_press:
            print('Keyboard Interrupt')
            break

        for i in range(len(particles)):
            if particles[i].best_fitness[-1]<best_global_fitness[-1] and check_validity_IQR(particles[i].best_fitness[-1], Q1_from_experiments, Q3_from_experiments, threshold_from_experiments):
                best_particle=particles[i]
                best_global_position.append(particles[i].position[-1])
                best_global_fitness.append(particles[i].best_fitness[-1])

        print('Best fitness score in iteration', it,'=',best_global_fitness[-1])

    return best_particle, best_global_fitness

if __name__=='__main__':
    main()