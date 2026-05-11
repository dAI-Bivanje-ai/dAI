import numpy as np
from src.visualization.data_visualizer import pripravi_pakete, sestavi_podatke
from data_logger import DataLogger
from windower import window_signal_seconds
from stft import compute_spectrograms

def load_session(bin_file):
    """
    Prebere eno .bin datoteko.
    Vrne (Fvz_acc, sig_acc, Fvz_gyro, sig_gyro).
    sig_acc.shape  = (N, 3)load_
    sig_gyro.shape = (N, 3)
    """
    
    logger = DataLogger()
    raw_packets = logger.parse_file(bin_file)
    
    packets = pripravi_pakete(raw_packets)
    
    fvz_gyro, gyro_raw = sestavi_podatke(packets, 1)
    fvz_acc, acc_raw = sestavi_podatke(packets, 2)
    
    sig_gyro = gyro_raw * 8.75e-3
    sig_acc = acc_raw * 1e-3
    #print(sig_gyro.shape)
    return (fvz_acc, sig_acc, fvz_gyro, sig_gyro)


def build_dataset(files):
    """
    Gre čez vse datoteke, za vsako pokliče load_session,
    windower in stft, doda labele.
    Vrne (X_acc, X_gyro, y).
    """
    all_acc = []
    all_gyro = []
    all_y = []
    
    for bin_file, label in files:
        fvz_acc, sig_acc, fvz_gyro, sig_gyro = load_session(bin_file)
        
        windows_acc = window_signal_seconds(sig_acc, fvz_acc)
        windows_gyro = window_signal_seconds(sig_gyro, fvz_gyro)
        
        spectograms_acc = compute_spectrograms(windows_acc)
        spectograms_gyro = compute_spectrograms(windows_gyro)
        
        M = min(spectograms_acc.shape[0], spectograms_gyro.shape[0])
        
        spectrograms_acc = spectograms_acc[:M]
        spectrograms_gyro = spectograms_gyro[:M]
        
        labels = np.full(M,label)
        
        all_acc.append(spectrograms_acc)
        all_gyro.append(spectrograms_gyro)
        all_y.append(labels)
    
    X_acc  = np.concatenate(all_acc,  axis=0)
    X_gyro = np.concatenate(all_gyro, axis=0)
    y      = np.concatenate(all_y,    axis=0)
    
    return X_acc, X_gyro, y
    
    
    
def save_dataset(X_acc, X_gyro, y, filename="dataset.npz"):
    """
    Shrani dataset na disk.
    """
    np.savez(filename, X_acc=X_acc, X_gyro=X_gyro, y=y)

