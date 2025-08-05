import sys
import os
import pandas as pd
import numpy as np
from utils import simulate_population_t_eval_multiprocessing, get_spiking_times
import da
import stg
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def get_resource_path(relative_path):
    """ Get the absolute path to a resource in a PyInstaller bundle or script directory. """
    try:
        # PyInstaller creates a temporary folder during execution and stores
        # bundled files there.
        base_path = sys._MEIPASS
    except Exception:
        # When running in normal Python mode (not bundled), use the script directory
        base_path = os.path.dirname(os.path.abspath(__file__))
        
    return os.path.join(base_path, '..', relative_path)


def main():
    csv_file = sys.argv[1]
    neuron_type = sys.argv[2]
    num_cpus = int(sys.argv[3])
    output_file = sys.argv[4]
    selected_ids = sys.argv[5].split(",")
    T = float(sys.argv[6])
    dt = float(sys.argv[7])
    t_transient = float(sys.argv[8])
    save_full_traces = sys.argv[9].lower() == 'true'

    # Your simulation code here...
    print(f"Running simulation for {neuron_type} with {num_cpus} CPUs; T = {T}, dt = {dt}")
    print(f"Selected IDs: {selected_ids}")

    selected_ids = [float(i) for i in selected_ids]

    # 1. We load the csv file
    if os.path.exists(csv_file):
        data = pd.read_csv(csv_file)
        print(f"Loaded {len(data)} rows from {csv_file}")
    else:
        print(f"File {csv_file} does not exist.")
        return
    
    # We will add a 'simulation' column to the dataframe, should be 'np.nan' by default
    data['simulation_V'] = np.nan
    t_eval = np.arange(t_transient, T+dt, dt)

    data = data.reset_index(drop=True)

    data_ids = data[data['ID'].isin(selected_ids)]
    data_idx = data[data['ID'].isin(selected_ids)].index
    
    print(f"Selected {len(data_ids)} rows for simulation")

    if neuron_type == "STG": # g_Na,g_Kd,g_CaT,g_CaS,g_KCa,g_A,g_H,g_leak
        conductances = ['g_Na', 'g_Kd', 'g_CaT', 'g_CaS', 'g_KCa', 'g_A', 'g_H', 'g_leak']
        simulate_ind = stg.simulate_individual_t_eval
        # population is N x 8 array
        population = np.array([data_ids[g].values for g in conductances]).T
        u0 = stg.get_default_u0()
        params = stg.get_default_parameters()

    elif neuron_type == "DA": # g_Na,g_Kd,g_CaL,g_CaN,g_ERG,g_NMDA,g_leak
        conductances = ['g_Na', 'g_Kd', 'g_CaL', 'g_CaN', 'g_ERG', 'g_NMDA', 'g_leak']
        simulate_ind = da.simulate_individual_t_eval
        # population is N x 7 array
        population = np.array([data_ids[g].values for g in conductances]).T
        u0 = da.get_default_u0()
        params = da.get_default_parameters()
    else:
        print(f"Unknown neuron type: {neuron_type}")
        return
    
    # remove any warning:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = simulate_population_t_eval_multiprocessing(
            simulation_function=simulate_ind,
            population=population,
            u0 = u0,
            t_eval=t_eval,
            params=params,
            max_workers=num_cpus,
            verbose=True,
            use_tqdm=False
        )

    print("Simulation completed, processing results...", flush=True)

    # We add the results to the dataframe
    for i in range(len(data_ids)):
        print(f"Post Processing {i+1}/{len(data_ids)}", flush=True)
        if save_full_traces:
            # Save the full trace as a string representation of the list
            data.at[data_idx[i], 'simulation_V'] = str(results[i][1].tolist())
        else:
            # Save only the spike times
            data.at[data_idx[i], 'spiking_times'] = str(get_spiking_times(t_eval, results[i][1])[1].tolist())

    # Finally we save the new version of the csv file into output_file
    data.to_csv(output_file, index=False)
    print("Simulation results are available !")

    print("RESULTS_READY")

if __name__ == "__main__":
    main()
