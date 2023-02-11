import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt
import siren

import time
import wget

import scipy.io.wavfile as wavfile
import io
from IPython.display import Audio

if not os.path.exists('gt_bach.wav'):
    url = "https://vsitzmann.github.io/siren/img/audio/gt_bach.wav"
    wget.download(url) 


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

class AudioFile(torch.utils.data.Dataset):
    def __init__(self, filename):
        self.rate, self.data = wavfile.read(filename)
        self.data = self.data.astype(np.float32)
        self.timepoints = get_mgrid(len(self.data), 1)

    def get_num_samples(self):
        return self.timepoints.shape[0]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        amplitude = self.data
        scale = np.max(np.abs(amplitude))
        amplitude = (amplitude / scale)
        amplitude = torch.Tensor(amplitude).view(-1, 1)
        return self.timepoints, amplitude



bach_audio = AudioFile('gt_bach.wav')

dataloader = DataLoader(bach_audio, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)

# Note that we increase the frequency of the first layer to match the higher frequencies of the
# audio signal. Equivalently, we could also increase the range of the input coordinates.
audio_siren = siren.Siren(in_features=1, out_features=1, hidden_features=64, 
                    hidden_layers=3, first_omega_0=3000, outermost_linear=True)
audio_siren.cuda()

rate, _ = wavfile.read('gt_bach.wav')

model_input, ground_truth = next(iter(dataloader))
Audio(ground_truth.squeeze().numpy(),rate=rate)

total_steps = 1000 
steps_til_summary = 100

optim = torch.optim.Adam(lr=1e-4, params=audio_siren.parameters())

model_input, ground_truth = next(iter(dataloader))
model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

for step in range(total_steps):
    model_output, coords = audio_siren(model_input)    
    loss = F.mse_loss(model_output, ground_truth)
    
    if not step % steps_til_summary:
        print("Step %d, Total loss %0.6f" % (step, loss))
    
        if step == 0:
            fig, axes = plt.subplots(1,2)
            axes[0].plot(coords.squeeze().detach().cpu().numpy(),model_output.squeeze().detach().cpu().numpy())
            axes[1].plot(coords.squeeze().detach().cpu().numpy(),ground_truth.squeeze().detach().cpu().numpy())            
            # plt.show()
            
    elif step == total_steps-1:
        print("Step %d, Total loss %0.6f" % (step, loss))
        fig, axes = plt.subplots(1,2)
        axes[0].plot(coords.squeeze().detach().cpu().numpy(),model_output.squeeze().detach().cpu().numpy())
        axes[1].plot(coords.squeeze().detach().cpu().numpy(),ground_truth.squeeze().detach().cpu().numpy())        
        plt.show()

    optim.zero_grad()
    loss.backward()
    optim.step()
    
    
final_model_output, coords = audio_siren(model_input)

from torchsummary import summary as summary 
print('input size',len(model_input[0])) 
summary(audio_siren, (len(model_input[0]),1))  

# for name, param in audio_siren.named_parameters():
#     print(f'name:{name}')
#     print(type(param))
#     print(f'param.shape:{param.shape}')
#     print(f'param.requries_grad:{param.requires_grad}')
#     print('------------------------------------------')

wavfile.write('gt_bach_siren.wav',rate,final_model_output.cpu().detach().squeeze().numpy())

# for param in audio_siren.parameters():
#     print(param)
#     print(param.size())
    

        
  
# for name, child in audio_siren.named_children():
#     for param in child.parameters():
#         print(name, param)    
# from scipy.io.wavfile import write
# import numpy as np
# samplerate = 44100; fs = 100
# t = np.linspace(0., 1., samplerate)
# amplitude = np.iinfo(np.int16).max
# data = amplitude * np.sin(2. * np.pi * fs * t)
# write("example.wav", samplerate, data.astype(np.int16))
# Audio(final_model_output.cpu().detach().squeeze().numpy(),rate=rate)
