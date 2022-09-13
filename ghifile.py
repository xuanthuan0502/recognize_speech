import numpy as np
from scipy.fftpack import dct
import scipy.io.wavfile
import matplotlib.pyplot as plt
import os
import hashlib
import re
import math


def get_features(file_path, pre_emphasis = 0.95, frame_size = 0.025, frame_step = 0.01, NFFT = 512,\
                 nfilt = 26, low_freq_hz = 300, num_ceps = 13):
    
    """
    Args:
        file_path: File path of the data sample.
        pre_emphasis: filter coefficient for pre emphasis phase.
        frame_size: size of the frames in framing phase.
        frame_step: size of the overlap in framing phase.
        NFFT: point numbers of discrete Fourier Transform (DFT).
        nfilt: number of filters used in filter Banks calulation phase.
        low_freq_hz: lower frequency used in filter Banks calulation phase.
        num_ceps: number of Cepstral Coefficients. 

    Returns:
        numpy array, of features: MFFCCs and delta coefficients.
    """
    sample_rate, signal = scipy.io.wavfile.read(file_path)
    #sample_rate = 16000    
    
    #Preemphasis
    signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[: -1])
    #plotSignal(time, signal)

    #Framing
    signal_length = len(signal)
    frame_length = int(frame_size * sample_rate)
    step_length = int(frame_step * sample_rate)


    num_frames = int(np.ceil(float(np.abs(signal_length-frame_length))/step_length))+1
    
    pad_signal_length = (frame_length + num_frames * step_length) - signal_length
    pad_signal = np.zeros(pad_signal_length)
    signal = np.append(signal,pad_signal)

    indices_matrix = np.tile(np.arange(0,frame_length),(num_frames,1))
    offset_indices = np.arange(0,step_length*num_frames,step_length)
    indices_matrix = (indices_matrix[0:].T + (offset_indices[0:])).T
    frames = signal[indices_matrix.astype(np.int32, copy=False)]

    #Windowing
    #Explicit implementation:
    #w = np.arange(0,frame_length)
    #w = 0.54 - 0.46 * np.cos((2 * np.pi * w) / (frame_length - 1))
    #frames*=w
    
    frames *= np.hamming(frame_length)
    
    #Discrete Fourier Transformation
    magnitude_frames = np.absolute(np.fft.rfft(frames, NFFT))
    
    #Power spectrum
    pow_frames = (magnitude_frames ** 2) / NFFT
    
    #Compute energy
    energy = np.sum(pow_frames, axis = 1)

    #Filter Banks
    # nfilt = filters number
    # low_freq_hz = 300 usually default is 0 (set to 300 for discard too low frequency,\
    # likely generated from noise)
    highfreq = sample_rate / 2
    low_freq_mel = (2595 * np.log10(1 + low_freq_hz / 700.))  # Convert Hz to Mel
    high_freq_mel = (2595 * np.log10(1 + highfreq / 700.))  # Convert Hz to Mel

    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2 ) # Equally spaced in Mel scale
    hz_points = (700 * (10** (mel_points / 2595.0) - 1 )) # Convert Mel to Hz
    
    #hz_points 28 gia tri : 300 -> 8000
    bin = np.floor((NFFT + 1) * hz_points / sample_rate) # our points are in Hz, but we use fft bins,\
                                                         # so we have to conver from Hz to fft bin number
    # print(bin)
  #  [  9.  12.  15.  18.  21.  25.  29.  33.  38.  43.  49.  54.  61.  68.
  #75.  84.  93. 102. 113. 124. 136. 150. 164. 180. 196. 215. 235. 256.]
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    #fbank.size=6682
   
    for j in range(0,nfilt):
            for i in range(int(bin[j]), int(bin[j+1])):
                fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
            for i in range(int(bin[j+1]), int(bin[j+2])):
                fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
   
    # Plot filterbank if you want           
    #plotFilterbank(fbank)         
    
    filter_banks = np.dot(pow_frames, fbank.T)
    
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # if energy is zero, we get problems with log
    
    filter_banks = np.log10(filter_banks)  # dB
    
    #Mel Frequency Cepstral Coefficients
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 0:(num_ceps)] # Keep 0 to num_ceps-1
    
    mfcc[:, 0] = np.log(energy + 1e-8)  # the zeroth cepstral coefficient is replaced with the log of\
                                        # the total frame energy


    cep_lifter = 22
    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift  

    #Mean normalization 
    filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)

    #Delta compute
    N = 2
    num_frames = mfcc.shape[0]
  
    denominator = 2 * sum([n**2 for n in range(1, N+1)])
    
    delta_feat = np.empty_like(mfcc)
    delta_feat2 = np.empty_like(mfcc)
    padded = np.pad(mfcc, ((N, N), (0, 0)), mode='edge')   # padded version of feature vectors(mfcc) (appending N*2 rows)
   # print(mfcc[1])
   # print(padded.size)
    for t in range(num_frames):
        delta_feat[t] = np.dot(np.arange(-N, N+1), padded[t : t+2*N+1]) / denominator   # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
  #  print(np.arange(-N, N+1)[0 : 5])
    #Append mfcc and Delta features
    features = np.append(mfcc, delta_feat, axis = 1)
    padded = np.pad(delta_feat, ((N, N), (0, 0)), mode='edge')
    for t in range(num_frames):
        delta_feat2[t] = np.dot(np.arange(-N, N+1), padded[t : t+2*N+1]) / denominator
    features = np.append(features, delta_feat2, axis = 1)     
    #print(features.size/24)
    
    print(features.size) 
    return features
def main() :
    path_dataset = "Dataset/data/data_train"
    train_dict_fruit = {}
    test_dict_fruit = {}
    labels_list = []
    features_list = []
    
    
    for root_dir, sub_dir, file in os.walk(path_dataset):
        sub_dir[:] = [d for d in sub_dir ]
        for wave in file:
            if(re.match('.*\.wav$',wave)):
              #  print(wave)
                file_path = (os.path.join(root_dir, wave))
                label = os.path.relpath(root_dir, path_dataset)
                feature = get_features(file_path)
                labels_list.append(label)
                features_list.append(feature)          
                np.savetxt("Dataset/features/data_train/"+label+"/"+wave+".txt", feature, delimiter =", ")

    
              
   

main()