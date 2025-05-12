import pandas as pd
import numpy as np
import multiprocessing as mp
import time

class InferenceWrapper:
    
    @staticmethod
    def default_postprocessing(results):
        results['g_u'] = results['g_u'].clip(lower=0)
        return results
    
    @staticmethod
    def cleaner_default_postprocessing(results, rounding=True):
        results['g_u'] = results['g_u'].clip(lower=0)
        results.loc[results['label'] == 0, 'f_spiking'] = 0
        results.loc[results['label'] == 1, 'f_spiking'] = 0
        results.loc[results['label'] == 3, 'f_spiking'] = 0
        results.loc[results['label'] == 0, 'f_intra_bursting'] = 0
        results.loc[results['label'] == 1, 'f_intra_bursting'] = 0
        results.loc[results['label'] == 2, 'f_intra_bursting'] = 0
        results.loc[results['label'] == 0, 'f_inter_bursting'] = 0
        results.loc[results['label'] == 1, 'f_inter_bursting'] = 0
        results.loc[results['label'] == 2, 'f_inter_bursting'] = 0
        results.loc[results['label'] == 0, 'duration_bursting'] = 0
        results.loc[results['label'] == 1, 'duration_bursting'] = 0
        results.loc[results['label'] == 2, 'duration_bursting'] = 0
        results.loc[results['label'] == 0, 'nbr_spikes_bursting'] = 0
        results.loc[results['label'] == 1, 'nbr_spikes_bursting'] = 0
        results.loc[results['label'] == 2, 'nbr_spikes_bursting'] = 0
        if rounding:
            results['nbr_spikes_bursting'] = results['nbr_spikes_bursting'].apply(lambda x: round(x))
        return results
    
    @staticmethod
    def std_postprocessing(results):
        results['std_g_s'] = results['logvar_g_s'].apply(lambda x: np.sqrt(np.exp(x)))
        results['std_g_u'] = results['logvar_g_u'].apply(lambda x: np.sqrt(np.exp(x)))
        results.drop(columns=['logvar_g_s', 'logvar_g_u'], inplace=True)
        return results
            
    @staticmethod
    def identity_postprocessing(results):
        return results
    
    @staticmethod
    def cleaner_and_std_postprocessing(results, rounding=True):
        r = InferenceWrapper.cleaner_default_postprocessing(results, rounding=rounding)
        r = InferenceWrapper.std_postprocessing(r)
        return r
    
class SpikeFeatureExtractor:
    def __init__(self, postprocessing=None, model="stg"):
        self.postprocessing = postprocessing
        if postprocessing is None:
            self.postprocessing = InferenceWrapper.identity_postprocessing
        self.model = model

    def extract_from_csv(self, csv_path, save_path=None, verbose=False, num_workers=8):
        if verbose:
            print(f"Loading data from {csv_path}", flush=True)
        
        with open(csv_path, 'r') as f:
            header = f.readline().strip().split(',')

        column_to_load = ['spiking_times']
        if 'ID' in header:
            column_to_load.append('ID')

        data = pd.read_csv(csv_path, usecols=column_to_load)

        if verbose:
            print(f"Data loaded, number of samples: {len(data)}", flush=True)
        
        results = self.extract_from_dataframe(data, verbose=verbose, num_workers=num_workers)

        if save_path is not None:
            if verbose:
                print(f"Saving results to {save_path}", flush=True)
            results.to_csv(save_path, index=False)
        else:
            if verbose:
                print("No save path provided, returning the results.", flush=True)
        return results
    
    def extract_from_dataframe(self, data, verbose=False, num_workers=16, should_preprocess=True):
        results = pd.DataFrame(columns=[
            'spiking_times', 'label', 'f_spiking', 'f_intra_bursting', 'f_inter_bursting', 'duration_bursting', 'nbr_spikes_bursting', 'ID'
        ], index=data.index)
        
        results["spiking_times"] = data["spiking_times"].copy()
        if 'ID' in data.columns:
            results["ID"] = data["ID"].copy()
        else:
            results["ID"] = np.arange(len(data))
            results["ID"] = results["ID"].astype('int32')
            print("No ID column found, creating a default ID column. We strongly recommend to add an ID column to your data.", flush=True)

        if should_preprocess:
            if verbose:
                print("Preprocessing data", flush=True)
            
            results['spiking_times'] = results['spiking_times'].apply(
                lambda x: np.fromstring(x[1:-1], sep=',').astype(np.float32) 
                if pd.notna(x) and x != '[]' 
                else (np.array([]) if x == '[]' else np.nan)
            )
            results['spiking_times'] = results['spiking_times'].apply(
                lambda x: x - x[0] if len(x) > 0 else x
            )   
            
            if verbose:
                print("Data preprocessed, extracting features", flush=True)

        chunks = []
        for i in range(0, num_workers):
            start = i * len(results) // num_workers
            end = (i + 1) * len(results) // num_workers
            end = min(end, len(results))
            chunks.append(results.iloc[start:end])

        if verbose:
            print(f"Extracting features from {len(results)} samples using {num_workers} workers", flush=True)
        
        extracted = []
        with mp.Pool(num_workers) as pool:
            for i, chunk_result in enumerate(pool.imap(self.parallel_extract_features, chunks)):
                extracted.append(chunk_result)
                if verbose:
                    print(f"[{time.strftime('%H:%M:%S')}] Processed {i + 1}/{num_workers} chunks", flush=True)

        return pd.concat(extracted)
    
    def parallel_extract_features(self, data):
        return data.apply(self.extract_from_row, axis=1)

    def extract_from_row(self, row):
        row['f_spiking'] = 0
        row['f_intra_bursting'] = 0
        row['f_inter_bursting'] = 0
        row['duration_bursting'] = 0
        row['nbr_spikes_bursting'] = 0

        spiking_times = row['spiking_times']

        label = -1
        if len(spiking_times) < 3:
            label = 0  # Silent
    
        cv_th = 0.15 if self.model == "stg" else 0.15

        if label == -1:
            ISIs = np.diff(spiking_times)
            CV_ISI = np.std(ISIs) / np.mean(ISIs)
            if CV_ISI <= cv_th:
                label = 1  # Regular spiking
            else:
                label = 2  # Bursting

        row['label'] = label

        if row['label'] == 1:
            if len(spiking_times) > 0:
                ISIs = np.diff(spiking_times)
                if len(ISIs) > 0:
                    row['f_spiking'] = (1000. / ISIs).mean()

        elif row['label'] == 2:
            if len(spiking_times) > 1:
                ISIs = np.diff(spiking_times)
                threshold = (ISIs.max() + ISIs.min()) / 2
                burst_starts = np.where(ISIs > threshold)[0] + 1
                bursts = np.split(spiking_times, burst_starts)
                bursts = [b for b in bursts if len(b) > 1]
                bursts = bursts[1:-1]

                if len(bursts) > 0:
                    row['nbr_spikes_bursting'] = np.nanmean([len(b) for b in bursts])
                    row['duration_bursting'] = np.nanmean([b[-1] - b[0] for b in bursts])
                    row['f_intra_bursting'] = np.nanmean([len(b) / (b[-1] - b[0]) if (b[-1] - b[0]) > 0 else np.nan for b in bursts]) * 1000.0
                    
                    burst_onsets = [b[0] for b in bursts]
                    if len(burst_onsets) > 1:
                        inter_burst_ISIs = np.diff(burst_onsets)
                        row['f_inter_bursting'] = 1000. / np.mean(inter_burst_ISIs)
                    else:
                        row['f_inter_bursting'] = np.nan
                else:
                    row['nbr_spikes_bursting'] = np.nan
                    row['duration_bursting'] = np.nan
                    row['f_intra_bursting'] = np.nan
                    row['f_inter_bursting'] = np.nan

        return row
