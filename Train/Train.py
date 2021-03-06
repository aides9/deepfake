"""
Created on Thu Dec 31 04:25:24 2020

@author: Jen
"""

import torch
import torch.nn as nn
import numpy as np
import time, os, copy
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler
from tqdm import tqdm_notebook as tqdm

class Train():
    
    def __init__(self, model, dataset, device, writer, config=None, utils=None, batch_size=16):
        self.model = model
        self.writer = writer
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.dataset_sizes = {}
        self.class_names = []
        self.config = config
        self.utils = utils
        self.feature_center = None
        
    def create_loader(self):

        # Creating data indices for training and validation splits:
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.2 * dataset_size))
        np.random.seed(322)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        
        # Creating loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        
        train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, 
                                                   sampler=train_sampler)
        validation_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size,
                                                        sampler=valid_sampler)
        dataloaders = {'train': train_loader,'val':validation_loader}
        dataset_sizes = {'train':len(train_loader.sampler),'val':len(validation_loader.sampler)}
        self.class_names = self.dataset.classes
        self.dataset_sizes = dataset_sizes
        
        return dataloaders
        
    def run(self, num_epochs=10, criterion=None, optimizer=None, scheduler=None):
      
        criterion = criterion or torch.nn.BCEWithLogitsLoss().to(self.device)
        optimizer = optimizer or optim.Adam(self.model.parameters(), lr=0.0001)
        scheduler = scheduler or lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        dataloaders = self.create_loader()

        self.model = self.model.to(self.device)
        since = time.time()
        
        # Set all parameters trainable
        for param in self.model.parameters():
            param.requires_grad = True
    
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        
        epoch_time = [] # we'll keep track of the time needed for each epoch
    
        for epoch in range(num_epochs):
            epoch_start = time.time()
            print('Epoch {}/{}'.format(epoch+1, num_epochs))

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode
    
                running_loss = 0.0
                running_corrects = 0
    
                # Iterate over data.
                for idx, (inputs, labels) in tqdm(enumerate(dataloaders[phase]),total=len(dataloaders[phase])):
                    inputs = inputs.to(self.device, non_blocking=True, dtype=torch.float)
                    labels = labels.to(self.device, non_blocking=True, dtype=torch.float)
    
                    # zero the parameter gradients
                    optimizer.zero_grad()
    
                    # Forward
                    # Track history if only in training phase
                    with torch.set_grad_enabled(phase == 'train'):

                        loss, preds = self.get_loss_preds(inputs, labels, criterion, phase)

                        # backward + optimize only if in training phase
                        if phase == 'train':
    
                            # Backward calculation
                            loss.backward()
                            # Update gradient in batch level
                            optimizer.step()
                            
                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.unsqueeze(1))
    
                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]
                
                self.writer.add_scalar('Loss/'+phase, epoch_loss, epoch)
                self.writer.add_scalar('Accuracy/'+phase, epoch_acc, epoch)
    
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                
                # Update gradient in epoch level
                scheduler.step()

                # Update attention feature if exist
                if self.feature_center is not None: 
                    self.feature_center*=2  
    
                # Deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
            
            # Add the epoch time
            t_epoch = time.time() - epoch_start
            epoch_time.append(t_epoch)
            print()
    
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        self.writer.flush()
        self.writer.close()
    
        # Load best model weights
        self.model.load_state_dict(best_model_wts)
        
        return self.model, dataloaders

    def get_loss_preds(self, inputs,labels,criterion, phase):
      
        if hasattr(self.model, 'CapsuleNet'):
           outputs, preds = self.model(inputs)
           loss = criterion(outputs, labels)
           preds = torch.unsqueeze(torch.round(preds), 1)
        
        elif hasattr(self.model, 'bap'):
          if (phase=='train'):
              if(self.feature_center is None):
                self.feature_center = torch.zeros(self.config.num_classes, self.config.num_attentions * self.model.num_features, device=self.device)

              y_pred_raw, feature_matrix, attention_map = self.model(inputs, dropout=True)
              feature_center_batch = F.normalize(self.feature_center[labels.type(torch.LongTensor)], dim=-1)
              self.feature_center[labels.type(torch.LongTensor)] += self.config.beta * (feature_matrix.detach() - feature_center_batch)
              
              with torch.no_grad():
                  crop_images = self.utils.batch_augment(inputs, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6), padding_ratio=0.1)
      
              y_pred_crop, _, _ = self.model(crop_images)
              with torch.no_grad():
                  drop_images = self.utils.batch_augment(inputs, attention_map[:, 1:, :, :], mode='drop', theta=(0.4, 0.7))

              y_pred_drop, _, _ = self.model(drop_images)
              center_loss = self.utils.CenterLoss().to(self.device)
              loss = criterion(y_pred_raw, labels.long()) + \
                  criterion(y_pred_crop, labels.long()) / 3. + \
                  criterion(y_pred_drop,labels.long()) / 2. + \
                  center_loss(feature_matrix, feature_center_batch)

              preds = torch.unsqueeze(torch.argmax(y_pred_raw, 1), 1)

          elif(phase=='val'):
              y_pred_raw, _, attention_map = self.model(inputs)
              crop_images = self.utils.batch_augment(inputs, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
              y_pred_crop, _, _ = self.model(crop_images)
              y_pred = (y_pred_raw + y_pred_crop) / 2.
              loss = criterion(y_pred, labels.long())
              preds = torch.unsqueeze(torch.argmax(y_pred, 1), 1)

        else :
           outputs = self.model(inputs)
           loss = criterion(outputs, labels.unsqueeze(1))
           preds = torch.round(torch.sigmoid(outputs))

        return loss, preds


    def get_classname():
        return self.class_names
    
    def get_dataset_size():
        return self.dataset_size


    



