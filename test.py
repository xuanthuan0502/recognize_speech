import numpy as np
from scipy.fftpack import dct
import scipy.io.wavfile
import matplotlib.pyplot as plt
import os
import hashlib
import re
from hmmlearn import hmm
import pickle
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support
from prettytable import PrettyTable
from prettytable import PLAIN_COLUMNS
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
   # print(pow_frames[0].size)
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
   # print(padded[1])
    for t in range(num_frames):
        delta_feat[t] = np.dot(np.arange(-N, N+1), padded[t : t+2*N+1]) / denominator   # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
  #  print(np.arange(-N, N+1)[0 : 5])
    #Append mfcc and Delta features
    features = np.append(mfcc, delta_feat, axis = 1)
    padded = np.pad(delta_feat, ((N, N), (0, 0)), mode='edge')
    for t in range(num_frames):
        delta_feat2[t] = np.dot(np.arange(-N, N+1), padded[t : t+2*N+1]) / denominator
    features = np.append(features, delta_feat2, axis = 1)     
   # print(features.size)
    return features
def main() :
    train_dict_word = {}
    test_dict_word = {}
    labels_list = []
    features_list = []
    path_dataset = "Dataset/features/data_train"
    for root_dir, sub_dir, file in os.walk(path_dataset):
            sub_dir[:] = [d for d in sub_dir ]
            for txt in file:
                 if(re.match('.*\.txt$',txt)):
                    file_path = (os.path.join(root_dir, txt))
                    label = os.path.relpath(root_dir, path_dataset)
                    feature = np.loadtxt("Dataset/features/data_train/"+label+"/"+txt,delimiter=', ')
                    labels_list.append(label)
                    features_list.append(feature)
                    
    words = np.unique(labels_list)
    """
    Split 85% for training and 15% for test
    """
    training_features = features_list
    training_labels = labels_list
    labels_list = []
    features_list = []
    path_dataset = "Dataset/data/data_set"
    for root_dir, sub_dir, file in os.walk(path_dataset):
        sub_dir[:] = [d for d in sub_dir  ]
        for wave in file:
           
            if(re.match('.*\.wav$',wave)):
                file_path = (os.path.join(root_dir, wave))
                label = os.path.relpath(root_dir, path_dataset)
                feature = get_features(file_path)
                labels_list.append(label)
                features_list.append(feature)   
    test_features = features_list
    test_labels = labels_list             
    for i in range(len(training_features)):
        if training_labels[i] not in train_dict_word:
            train_dict_word[training_labels[i]] = []
            train_dict_word[training_labels[i]].append(training_features[i])
        else:
            train_dict_word[training_labels[i]].append(training_features[i])            
    for i in range(len(test_features)):
        if test_labels[i] not in test_dict_word:
            test_dict_word[test_labels[i]] = []
            test_dict_word[test_labels[i]].append(test_features[i])
        else:
            test_dict_word[test_labels[i]].append(test_features[i])
    #Train dataset

    GMMHMM_models_word = {} # dict of HMMs (one model for each word into the dataset)
    num_states = 3 # States number of HMM
    num_mix = 2 # number of mixtures for each hidden state
    covariance_type = 'diag'  # covariance type
    num_iter = 10  # number of max iterations
    bakis_level = 2

    start_prob = np.zeros(num_states) # start probability prior
    start_prob[0:bakis_level - 1] = 1 / float(1 / (bakis_level - 1))

    trans_mat = np.eye(num_states) # transaction matrix probability prior 
    for i in range(num_states - (bakis_level - 1)):
        for j in range(bakis_level):
            trans_mat[i, i + j] = 1 / bakis_level

    for i in range((num_states - (bakis_level ) + 1), num_states ):
        trans_mat[i,i:] = (1 / (num_states - i))


    model_number = 0
    for word in train_dict_word:
        model = hmm.GMMHMM(n_components = num_states, n_mix = num_mix, startprob_prior = start_prob,\
                                transmat_prior = trans_mat, covariance_type = covariance_type,\
                                n_iter = num_iter, verbose=False)

        train_samples = train_dict_word[word]
        length_samples = np.zeros(len(train_samples), dtype=np.int) 
        for elem in range(len(train_samples)):
            length_samples[elem] = train_samples[elem].shape[0]
        
        train_samples = np.vstack(train_samples) # Stack arrays in train_samples in sequence vertically 

            
        
        
        
        #model.fit(train_samples, length_samples) # MODEL FIT
        model.fit(train_samples)
        
        GMMHMM_models_word[word] = model
        print("Finish train model GMM-HMM %s" % model_number)
        model_number += 1
    num_words = len(train_dict_word)
    print("Finish train %s GMM-HMMs for %s different words" % (num_words, num_words))


    trained_model_word = GMMHMM_models_word

    print("")

    #Test data

    score_count = 0
    words_number = 0
    y_true = []
    y_pred = []
    for word in test_dict_word.keys():
        test_samples = test_dict_word[word]
        for speech_word in test_samples:
            words_number += 1
            score_models = {}
            for word_model in trained_model_word.keys():
                model = trained_model_word[word_model]
                score = model.score(speech_word)
                score_models[word_model] = score
            predict_word = max(score_models, key = score_models.get)
            print(word, ": ", predict_word)
            y_true.append(word)
            y_pred.append(predict_word)
            if predict_word == word:
                score_count += 1

                
    accuracy = (100 * score_count / words_number)            
    print("Recognition rate %s" %(accuracy))
    #euclid
    score_count = 0
    words_number = 0
    y_true = []
    y_pred = []
    for word in test_dict_word.keys():
        test_samples = test_dict_word[word]
        for test_word in test_samples:
            words_number += 1
            score_models = {}
            for train_model in train_dict_word.keys():
                feature_train = train_dict_word[train_model]
                #print(feature_train[0][0][0])
                dem = 0
                score = 0;
                for sample in feature_train :
                    total = 0
                    for i in range(len(sample)):
                        euclid = 0
                        for j in range(39):
                            euclid= euclid + (test_word[i][j]-sample[i][j])**2
                        euclid = math.sqrt(euclid)
                        total = total + euclid
                    score = score+total            
                
               # print(score)
                #10/99/24
                score_models[train_model] = score
            predict_word = min(score_models, key = score_models.get)
            print(word, ": ", predict_word)
            y_true.append(word)
            y_pred.append(predict_word)
            if predict_word == word:
                score_count += 1

                
    accuracy = (100 * score_count / words_number)            
    print("Recognition rate %s" %(accuracy))
    

main()