import numpy as np
from data_visualizer import pripravi_pakete, sestavi_podatke
from data_logger import DataLogger
from windower import window_signal, window_signal_seconds

def load_session(bin_file):
    """
    Prebere eno .bin datoteko.
    Vrne (Fvz_acc, sig_acc, Fvz_gyro, sig_gyro).
    sig_acc.shape  = (N, 3)
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
    
    
    
    
    
    
