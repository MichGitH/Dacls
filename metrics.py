import librosa
import numpy as np
import torchaudio
import torch
from torch import Tensor
from typing import Union
from pystoi import stoi
from pesq import pesq
from speechmos import plcmos, dnsmos



#def _melspectrogram(x: np.ndarray) -> np.ndarray:
 #   return librosa.feature.melspectrogram(
  #      x,
 #       n_mels=64,
 #       sr=32000,
 #       n_fft=512,
 #       power=1,
#        hop_length=256,
 #       win_length=512,
 #       window='hann'
 #   )



def _melspectrogram(x: np.ndarray) -> np.ndarray:
    return librosa.feature.melspectrogram(
        y=x,  # Pass x as a keyword argument
        n_mels=64,
        sr=32000,
        n_fft=512,
        power=1,
        hop_length=256,
        win_length=512,
        window='hann'
    )


def mel_spectral_convergence(y_pred: Union[Tensor, np.ndarray], y_true: Union[Tensor, np.ndarray]) -> np.ndarray:
    assert type(y_pred) == type(
        y_true), f'y_pred and y_true must be of the same type. Found {type(y_pred)} (y_pred) and {type(y_true)} (y_true).'

    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()

    mel_true = _melspectrogram(y_true)
    mel_pred = _melspectrogram(y_pred)

    mel_sc = np.linalg.norm(mel_pred - mel_true, ord="fro") / np.linalg.norm(mel_true, ord="fro")

    return mel_sc


def nmse(y_pred: Union[Tensor, np.ndarray], y_true: Union[Tensor, np.ndarray]) -> np.ndarray:
    assert type(y_pred) == type(
        y_true), f'y_pred and y_true must be of the same type. Found {type(y_pred)} (y_pred) and {type(y_true)} (y_true).'

    if isinstance(y_pred, Tensor):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()

    nmse_db = 20 * np.log10(np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true))

    return nmse_db


#def calculate_pesq(y_ref, y_pred, sr):
    #""" Calcola PESQ (Perceptual Evaluation of Speech Quality) """
   # return pesq(sr, y_ref, y_pred, 'wb')  # 'wb' indica wideband
def calculate_pesq(y_ref, y_pred, sr):
    """ Calcola PESQ (Perceptual Evaluation of Speech Quality) """
    # Controllo se i segnali non sono vuoti o silenziosi
    if np.max(np.abs(y_ref)) == 0 or np.max(np.abs(y_pred)) == 0:
        raise ValueError("Uno dei segnali è completamente silenzioso.")
    if len(y_ref) == 0 or len(y_pred) == 0:
        raise ValueError("Uno dei segnali è vuoto.")
    
    # Allinea la lunghezza dei segnali
    min_len = min(len(y_ref), len(y_pred))
    y_ref = y_ref[:min_len]
    y_pred = y_pred[:min_len]

    # Verifica che la frequenza di campionamento sia supportata
    if sr not in [8000, 16000]:
        raise ValueError("PESQ supporta solo frequenze di campionamento di 8000 Hz o 16000 Hz.")
    

    return pesq(sr, y_ref, y_pred, 'wb')  # 'wb' indica wideband
    

def calculate_stoi(y_ref, y_pred, sr):
    """ Calcola STOI (Short-Time Objective Intelligibility) """
    return stoi(y_ref, y_pred, sr, extended=False)




def calculate_dnsmos(y_pred, sr):
    audio_max = np.max(np.abs(y_pred))
    if audio_max > 0:
        y_pred = y_pred / audio_max
    
    # Esegui DNSMOS e restituisci solo 'ovrl_mos'
    result = dnsmos.run(y_pred, sr)
    #print(result['ovrl_mos'])
    return result['ovrl_mos']

def calculate_plcmos(y_pred, sr):
    audio_max = np.max(np.abs(y_pred))
    if audio_max > 0:
        y_pred= y_pred / audio_max
    # Esegui DNSMOS e restituisci solo 'ovrl_mos'
    result1 = plcmos.run(y_pred, sr)
    #print(result1['plcmos'])
    return result1['plcmos']


# Funzione per calcolare gli MFCC (Mel-cepstrum coefficients) usando torchaudio
"""def calculate_mfcc(waveform: Union[Tensor, np.ndarray], sample_rate: int, n_mfcc: int = 13) -> np.ndarray:
    # Se waveform è un Tensor, lo converto in numpy array
    if isinstance(waveform, Tensor):
        waveform = waveform.detach().cpu().numpy()

    # Calcolo gli MFCC
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": 512,
            "n_mels": 64,
            "hop_length": 256,
            "mel_scale": "htk"
        }
    )
    mfcc_features = mfcc_transform(torch.from_numpy(waveform))
    return mfcc_features.numpy()"""
    

def calculate_mfcc(waveform, sr):
    """ Calcola il mel-cepstrum (MFCC) di un segnale audio """
    # Definizione della trasformazione MFCC
    max_amplitude = np.max(np.abs(waveform))
    
    if max_amplitude > 0:
        waveform= waveform / max_amplitude


    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sr, 
        n_mfcc=40
    )
    
    # Conversione dell'array NumPy in un tensore torch se necessario
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)

    # Calcolo delle caratteristiche MFCC
    mfcc_features = mfcc_transform(waveform)

    window_size=3
    smoothed_mfcc = np.apply_along_axis(
        lambda m: np.convolve(m, np.ones(window_size)/window_size, mode='valid'), 
        axis=1, 
        arr=mfcc_features
    )
    mfcc_mean = np.mean(smoothed_mfcc, axis=1)  # Media lungo l'asse temporale

    # Restituisce la media di tutti i coefficienti MFCC (singolo valore per riassumere)
    mfcc_mean = mfcc_mean.mean().item()

    
    return mfcc_mean
    



