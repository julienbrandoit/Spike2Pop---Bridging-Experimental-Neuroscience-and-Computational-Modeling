import sys
import torch
from inference import SpikeFeatureExtractor
import json
import base_model
import adapted_model
from torch.nn.utils.rnn import pad_sequence
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import stg
import da
import pandas as pd
import numpy as np
from multiprocessing import Pool
import time
import os

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


def load_model(model_name, device):
    # load the config json file with the path and the config
    config_path = get_resource_path('config.json')
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            if not isinstance(config, dict):
                raise ValueError("Loaded config is not a dictionary.")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)

    if model_name == "STG":
        
        model_path = config['base_model_weights_path']
        config_path_m = config['base_model_args_path']
        config_path_m = get_resource_path(config_path_m)
        model_path = get_resource_path(model_path)
        vth = config['base_model_v_th']

        # load the config json file with the path and the config
        try:
            with open(config_path_m, 'r') as f:
                config = json.load(f)
                if not isinstance(config, dict):
                    raise ValueError("Loaded config is not a dictionary.")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error loading config file: {e}")
            sys.exit(1)



        # SETUP MODEL
        model = base_model.DICsNet(
            d_encoder=config['d_encoder'],
            n_heads=config['n_heads'],
            dropout=config['dropout'],
            n_blocks_encoder=config['n_blocks_encoder'],
            n_blocks_decoder=config['n_blocks_decoder'],
            d_latent=config['d_latent'],
            activation=config['activation'],
            inference_only=False,
            should_log=config['should_log'],
        ).to(device)

        # Load the model state dict
        d = torch.load(model_path, map_location=device)
        d = d['model_state_dict']

        # load the model and print if anything is missing
        missing_keys, unexpected_keys = model.load_state_dict(d, strict=False)
        if len(missing_keys) > 0:
            print(f"Missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"Unexpected keys: {unexpected_keys}")

        model = model.to(device)
        # set the model to evaluation mode
        model.eval()

        return model, vth

    elif model_name == "DA":     
        model_path = config['da_adapters_weights_path']
        model_path = get_resource_path(model_path)
        base_model_path = config['base_model_weights_path']
        base_model_path = get_resource_path(base_model_path)
        config_path_m = config['da_adapters_args_path']
        config_path_m = get_resource_path(config_path_m)
        vth = config['da_adapters_v_th']

        # load the config json file with the path and the config
        try:
            with open(config_path_m, 'r') as f:
                config = json.load(f)
                if not isinstance(config, dict):
                    raise ValueError("Loaded config is not a dictionary.")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error loading config file: {e}")
            sys.exit(1)

        model = adapted_model.DICsNet(
            d_encoder=config['d_encoder'],
            n_heads=config['n_heads'],
            dropout=config['dropout'],
            n_blocks_encoder=config['n_blocks_encoder'],
            n_blocks_decoder=config['n_blocks_decoder'],
            d_latent=config['d_latent'],
            activation=config['activation'],
            inference_only=False,
            should_log=config['should_log'],
            r=config['r_value']
        ).to(device)

        m = torch.load(base_model_path, map_location=device)
        adapted_model.map_to_lora_model(model, m['model_state_dict'])

        model.load_lora_adapter(model_path)

        model = model.to(device)

        model.eval()
        return model, vth
    else:
        raise ValueError(f"Unknown neuron name: {model_name}")

def forward_model(model, data, device, batch_size=512):
    # we loop over the data, build the tensor, put on device and forward
    model.eval()
    # create the columns for g_s_hat and g_u_hat
    data['g_s_hat'] = 0.0
    data['g_u_hat'] = 0.0
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            start_idx = i
            end_idx = min(i + batch_size, len(data))
            batch = data.iloc[start_idx:end_idx]
            spike_times = batch['spiking_times'].values
            L = batch['L'].values

            # Limit the spike_times to a maximum length of 250 (not mandatory but for the stg and the da it is highly sufficient)
            spike_times = [torch.tensor(x[:250], dtype=torch.float32) if len(x) > 250 else torch.tensor(x, dtype=torch.float32) for x in spike_times]
            L = [len(x) if len(x) < 250 else 250 for x in spike_times]

            spike_times = pad_sequence(spike_times, batch_first=True, padding_value=0)
            spike_times = spike_times.to(device)
            L = torch.tensor(L, dtype=torch.float32).cpu()

            # forward the model
            y_hat, y_hat_aux_c, y_hat_aux_m, y_hat_aux_s = model.forward_auxilliary(spike_times, L)
            g_s_hat = y_hat[:, 0]
            g_u_hat = y_hat[:, 1]

            data.iloc[start_idx:end_idx, data.columns.get_loc('g_s_hat')] = g_s_hat.cpu().numpy()
            data.iloc[start_idx:end_idx, data.columns.get_loc('g_u_hat')] = g_u_hat.cpu().numpy()

    return data

def process_row_stg(row):
        population_size, vth, row = row
        ID = row['ID']
        g_s_hat = row['g_s_hat']
        g_u_hat = row['g_u_hat']
        population = stg.generate_neuromodulated_population(population_size, vth, g_s_hat, g_u_hat, iterations=5)
        population = np.hstack((np.full((len(population), 1), ID), population))
        return population

def process_row_da(row):
    population_size, vth, row = row
    ID = row['ID']
    g_s_hat = row['g_s_hat']
    g_u_hat = row['g_u_hat']
    population = da.generate_neuromodulated_population(population_size, vth, g_s_hat, g_u_hat)
    population = np.hstack((np.full((len(population), 1), ID), population))
    return population

def forward_dics(data, neuron_type, num_cpus, population_size, vth):
    def with_progress(index, total):
        if index % 25 == 0 or index == total - 1:
            print(f"[{time.strftime('%H:%M:%S')}] Processed {index + 1}/{total} inputs", flush=True)

    if neuron_type == "STG":
        column_names = ['ID', 'g_Na', 'g_Kd', 'g_CaT', 'g_CaS', 'g_KCa', 'g_A', 'g_H', 'g_leak']
        processor = process_row_stg
    elif neuron_type == "DA":
        column_names = ['ID', 'g_Na', 'g_Kd', 'g_CaL', 'g_CaN', 'g_ERG', 'g_NMDA', 'g_leak']
        processor = process_row_da
    else:
        raise ValueError(f"Unknown neuron type: {neuron_type}")

    args = [(population_size, vth, row) for _, row in data.iterrows()]
    total = len(args)

    results = []
    with Pool(num_cpus) as pool:
        for i, result in enumerate(pool.imap(processor, args)):
            results.append(result)
            with_progress(i, total)

    # Efficient one-time concat
    df = pd.concat(
        [pd.DataFrame(pop, columns=column_names) for pop in results],
        ignore_index=True
    )

    dtype_map = {col: 'float32' for col in column_names if col != 'ID'}
    df = df.astype(dtype_map)

    return df

def main():
    if len(sys.argv) != 7:
        print("Usage: script.py <csv_file> <neuron_type> <num_cpus> <output_file> <use_gpu> <population_size>")
        sys.exit(1)

    csv_file = sys.argv[1]
    neuron_type = sys.argv[2]
    num_cpus = int(sys.argv[3])
    output_file = sys.argv[4]
    use_gpu = sys.argv[5] == "True"
    population_size = int(sys.argv[6])  # New argument

    print(f"Processing {csv_file} with {neuron_type} using {num_cpus} CPUs and generating population of size {population_size}.")
    if use_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using GPU")
        else:
            device = torch.device("cpu")
            print("GPU not available, using CPU instead")
    else:
        device = torch.device("cpu")
        print("Using CPU")
   
    print("Loading model...")
    model, vth = load_model(neuron_type, device)
    print("Model loaded successfully.")

    sys.stdout.flush()

    print("Preprocessing data...")
    sfe = SpikeFeatureExtractor()
    r = sfe.extract_from_csv(csv_file, verbose=True, num_workers=num_cpus)
    r = r[['spiking_times', 'label', 'ID']]

    sys.stdout.flush()


    r_data = r[r['label'] != 0].copy()
    r_data['L'] = r_data['spiking_times'].apply(lambda x: len(x))    
    print("Data preprocessing completed.")


    print("Generating population...")
    r_data_dics = forward_model(model, r_data, device)
    r_data_gbar = forward_dics(r_data_dics, neuron_type, num_cpus, population_size, vth)
    print("Population generation completed.")

    r_data_gbar.to_csv(output_file, index=False)
    print(f"Results ready to be saved !")
    print("RESULTS_READY")
    
if __name__ == "__main__":
    main()
