import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

import librosa
import librosa.display

import pygame
import pygame.mixer
from time import sleep

def Play_sound(filename):
    pygame.mixer.init()
    sound = pygame.mixer.Sound(filename)
    tmp = sound.play()
    while tmp.get_busy():
        pygame.time.delay(1)
        
def Wiener_filter(filename):
    
    (data, sampling_rate) = librosa.load(filename, sr=8000, mono = True)
    y = librosa.stft(data, hop_length=128, win_length=256) # 주파수 129등분, 1409프레임
    phase = np.angle(y) # 위상
    magnitude = np.abs(y) # 주파수
    
    a = np.zeros(shape = (1024, 1)) # 프레임 저장
    
    for i in range(8):
        
        for j in range(1024):
            
            a[j][0] = a[j][0] + np.abs(y[j][i])
    
    Navg = a / 8
    
    for k in range(1024):
        for l in range(len(magnitude[0])):
            magnitude[k][l] = magnitude[k][l] - Navg[k][0]
            
    Xkn = magnitude*np.exp(1j*phase)
    Xk = librosa.istft(Xkn, hop_length=128, win_length=256)
    
    return Xk 
    
