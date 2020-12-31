"""
Created on Thu Dec 31 04:25:24 2020

@author: Jen
"""

import cv2 as cv
import torchvision

class DFDC(torchvision.datasets.ImageFolder):
  def __init__(self, root, transform):
    super(DFDC, self).__init__(root, transform)

  def __getitem__(self, index):
    # override ImageFolder's method
    path, target = self.samples[index]
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    if self.transform is not None:
      sample = self.transform(image=img)['image']
    if self.target_transform is not None:
      target = self.target_transform(target)

    return sample, target


