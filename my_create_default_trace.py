import os
import math
import yaml
import librosa
import argparse
import numpy as np
from pathlib import Path

def create_trace(audio_test_path: Path, trace_folder: Path, packet_dim: int, sr: int, loss_rate: int) -> None:
    # Load the clean signal
    y_true, sr = librosa.load(audio_test_path, sr=sr, mono=True)

    # Simulate packet losses
    trace_len = math.ceil(len(y_true) / packet_dim)
    trace = np.zeros(trace_len, dtype=int)
    trace[np.arange(loss_rate, trace_len, loss_rate)] = 1
    #print(loss_rate)

    # Save trace
    if not os.path.exists(trace_folder):
        os.makedirs(trace_folder)

    np.save(trace_folder.joinpath(f'{audio_test_path.stem}.npy'), trace)

def process_directory(audio_dir: Path, trace_folder: Path, packet_dim: int, sr: int, loss_rate: int) -> None:
    # Iterate over all audio files in the directory
    for audio_file in audio_dir.glob('*.wav'):  # Assuming audio files have .wav extension
        create_trace(audio_file, trace_folder, packet_dim, sr, loss_rate)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-r', '--loss_rate', type=int, default=10)
    args = vars(parser.parse_args())

    # Read config.yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Read params from config file
    audio_dir = Path(config['path']['audio_test_path'])  # Now this should be the directory
    trace_folder = Path(config['path']['trace_dir'])
    packet_dim = int(config['global']['packet_dim'])
    sr = int(config['global']['sr'])

    # Process all files in the directory
    process_directory(audio_dir, trace_folder, packet_dim, sr, args['loss_rate'])
