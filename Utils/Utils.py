"""
Created on Thu Dec 31 04:25:24 2020

@author: Jen
"""
import torchvision
import numpy as np
import matplotlib.pyplot as plt

def gridshow(loader):
    inputs, classes = next(iter(loader))
    grid = torchvision.utils.make_grid(inputs)
    images = np.clip(grid.permute(1, 2, 0), 0, 1)
    plt.imshow(images)
    plt.pause(0.001) 
    return None

def view_capsule_activation(model,device,x,label):
    activation = {}
    
    def get_activation(name):
      def hook(model, input, output):
        activation[name] = output.detach().cpu()
      return hook

    capsule_layer = model.CapsuleNet.fea_ext.capsules
    for idx in range(len(capsule_layer)):
      capsule_layer[idx][3].register_forward_hook(get_activation('capsule'+str(idx)))
    
    _,preds = model(x.to(device))

    for key, value in activation.items():
      plt.figure(figsize=(30, 30))
      layer_viz = value[1, :, :, :]

      for i, filter in enumerate(layer_viz):
        plt.subplot(8, 8, i + 1)
        plt.imshow(filter, cmap='gray')
        plt.axis("off")
      print(f"Capsule: {key}, prediction: {preds[1]}, actual: {label[1]}")
      plt.show()
    return None