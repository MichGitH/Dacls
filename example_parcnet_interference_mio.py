import os
import yaml
import csv
import librosa
import numpy as np
import soundfile as sf
import pandas as pd
from pathlib import Path
from parcnet import PARCnet
from metrics import nmse, mel_spectral_convergence, calculate_stoi,calculate_pesq,calculate_dnsmos,calculate_plcmos,calculate_mfcc
from utils import simulate_packet_loss






def process_file(audio_test_path: Path, model_checkpoint: Path, trace_folder: Path, prediction_folder: Path, 
                 packet_dim: int, extra_dim: int, ar_order: int, diagonal_load: float, 
                 num_valid_ar_packets: int, num_valid_nn_packets: int, xfade_len_in: int, sr: int) -> dict:
    metrics_result = {'file_name': audio_test_path.stem}

    print(f"Processing file: {audio_test_path}")

    # Instantiate PARCnet
    parcnet = PARCnet(
        packet_dim=packet_dim,
        extra_dim=extra_dim,
        ar_order=ar_order,
        ar_diagonal_load=diagonal_load,
        num_valid_ar_packets=num_valid_ar_packets,
        num_valid_nn_packets=num_valid_nn_packets,
        model_checkpoint=model_checkpoint,
        xfade_len_in=xfade_len_in,
        device='cpu'
    )
    

    # Load the reference audio file
    y_ref, sr = librosa.load(audio_test_path, sr=sr, mono=True)

    # Load packet loss trace
    trace_path = trace_folder.joinpath(audio_test_path.stem).with_suffix('.npy')
    if not os.path.exists(trace_path):
        print(f"Trace file not found for {audio_test_path.stem}, skipping.")
        return metrics_result
    trace = np.load(trace_path)

    # Simulate packet losses
    y_lost = simulate_packet_loss(y_ref, trace, packet_dim)

    # Predict using PARCnet
    y_pred = parcnet(y_lost, trace)

    # Save wav files
    if not os.path.exists(prediction_folder):
        os.makedirs(prediction_folder)

    #sf.write(prediction_folder.joinpath(f"{audio_test_path.stem}.wav"), y_lost, sr)
    #sf.write(prediction_folder.joinpath(f"{audio_test_path.stem}__zero-filling.wav"), y_lost, sr)
    #sf.write(prediction_folder.joinpath(f"{audio_test_path.stem}__parcnet.wav"), y_pred, sr)
    sf.write(prediction_folder.joinpath(f"{audio_test_path.stem}.wav"), y_pred, sr)

    # Compute NMSE
    mask = np.repeat(trace, packet_dim).astype(bool)

    metrics_result['zero-filling-nmse_signal'] = nmse(y_lost, y_ref)
    metrics_result['zero-filling-nmse_packet'] = nmse(y_lost[mask], y_ref[mask])
    metrics_result['parcnet-nmse-signal'] = nmse(y_pred, y_ref)
    metrics_result['parcnet-nmse-packet'] = nmse(y_pred[mask], y_ref[mask])
    metrics_result['zero-filling-mel-sc_signal'] = mel_spectral_convergence(y_lost, y_ref)
    metrics_result['zero-filling-mel-sc_packet'] = mel_spectral_convergence(y_lost[mask], y_ref[mask])
    metrics_result['parcnet-mel-sc_signal'] = mel_spectral_convergence(y_pred, y_ref)
    metrics_result['parcnet-mel-sc_packet'] = mel_spectral_convergence(y_pred[mask], y_ref[mask])

    metrics_result['zero-filling-PESQ']= calculate_pesq(y_ref, y_lost, sr)
    metrics_result['parcnet-PESQ']= calculate_pesq(y_ref, y_pred, sr)
    metrics_result['zero-filling-STOI']= calculate_stoi(y_ref, y_lost, sr)
    metrics_result['parcnet-STOI']= calculate_stoi(y_ref, y_pred, sr)
    metrics_result['dnsmos-zero_filling'] = calculate_plcmos(y_lost, sr)
    metrics_result['dnsmos-parcnet'] = calculate_plcmos(y_pred, sr)
    metrics_result['plcmos-zero_filling'] = calculate_dnsmos(y_lost, sr)
    metrics_result['plcmos-parcnet'] = calculate_dnsmos(y_pred, sr)

    # Calcola gli MFCC
    metrics_result['mel-cepstra-zero-filling'] = calculate_mfcc(y_lost, sr)
    metrics_result['mel-cepstra-parcnet'] = calculate_mfcc(y_pred, sr)
    #print(f"MFCC shape: {mfcc_features.shape}")
   
    

    return metrics_result


def main():
    # ----------- Read config.yaml ----------- #
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Read paths from config file
    audio_dir = Path(config['path']['audio_test_path'])  # Directory containing audio files
    model_checkpoint = Path(config['path']['nn_checkpoint_path'])
    trace_folder = Path(config['path']['trace_dir'])
    prediction_folder = Path(config['path']['prediction_dir'])
    metrics_file = Path(config['path']['metrics_file'])  # Path for saving metrics

    # Read global params from config file
    sr = int(config['global']['sr'])
    packet_dim = int(config['global']['packet_dim'])
    extra_dim = int(config['global']['extra_pred_dim'])

    # Read AR params from config file
    ar_order = int(config['AR']['ar_order'])
    diagonal_load = float(config['AR']['diagonal_load'])
    num_valid_ar_packets = int(config['AR']['num_valid_ar_packets'])

    # Read NN params from config file
    num_valid_nn_packets = int(config['neural_net']['num_valid_nn_packets'])
    xfade_len_in = int(config['neural_net']['xfade_len_in'])

    metrics_list = []

    # Process each audio file in the directory
    for audio_file in audio_dir.glob('*.wav'):  # Adjust the file extension if needed
        metrics_result = process_file(
            audio_file, model_checkpoint, trace_folder, prediction_folder, 
            packet_dim, extra_dim, ar_order, diagonal_load, 
            num_valid_ar_packets, num_valid_nn_packets, xfade_len_in, sr
        )
        metrics_list.append(metrics_result)

    # Save metrics to a CSV file
    #if not metrics_file.exists():
        # Create CSV file and write headers
    with open('metrics/metrics_file.csv', mode='w', newline='') as file:
            
        writer = csv.writer(file)
        headers = metrics_list[0].keys()  # Extract headers from the first dictionary
        writer.writerow(headers)  # Write the header row
        for metrics in metrics_list:
            writer.writerow(metrics.values())  # Write data rows
     # Convert metrics to a pandas DataFrame
    
    df = pd.DataFrame(metrics_list)

    # Save metrics to an Excel file
    df.to_excel('metrics_file.xlsx', index=False)




    """else:
        # Append data to existing CSV file
        with open(metrics_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            for metrics in metrics_list:
                writer.writerow(metrics.values())  # Write data rows"""

    print(f"Metrics saved to {metrics_file}")
    print(f"Metrics saved to metrics_file.xlsx")

if __name__ == "__main__":
    main()
 