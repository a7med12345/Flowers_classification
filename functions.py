# Imports here
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets,models 
from collections import OrderedDict
import copy
from PIL import Image
import matplotlib.pyplot as plt

def pretrained_model_info(pretrained_name):
    
    #upload the pretrained model
    x="models."+pretrained_name+"(pretrained=True)"
    exec("model="+x)
    
    #Get keys and values of the model
    list_keys=[]
    list_value=[]
    for key in model.state_dict():
        value = model.state_dict().get(key)
        list_keys.append(key)
        list_value.append(value)
      
    #Get the last key of the model by eliminating chars after "."
    last_layer_key  = list_keys[-1].split('.')[0]
    
    #Get size of the input to the classification layer
    for key in list_keys:
        if(key.split('.')[0]==last_layer_key):
            break
            
    x= model.state_dict().get(key).size()
    
    #return name of the classifier layer and the size of its input
    return model,last_layer_key,int(x[1])

def building_model(hiden_units,pretrained_name):
    
    model,key,size = pretrained_model_info(pretrained_name)
    
    #Defining the classifier
    classifier =  nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(size, hiden_units)),
                          ('c_relu1', nn.ReLU()),
                          ('c_drop1', nn.Dropout()),
                          ('fc3', nn.Linear(hiden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    #fixing the gradient
    for param in model.parameters():
        param.requires_grad = False  
    
    #replacing the classifier
    x="model."+key+"=classifier"
    exec(x)
    return model

def data_preparation(data_dir,train_batch=32,valid_batch=16,test_batch=16):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    test_valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    # Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(root=train_dir,transform=train_transforms)
    valid_datasets = datasets.ImageFolder(root=valid_dir,transform=test_valid_transforms)
    test_datasets = datasets.ImageFolder(root=test_dir,transform=test_valid_transforms)


    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = DataLoader(train_datasets, batch_size=train_batch, shuffle=True)
    validloader = DataLoader(valid_datasets, batch_size=valid_batch)
    testloader = DataLoader(test_datasets, batch_size=test_batch)
    
    return train_datasets,trainloader,validloader,testloader


def train(model, trainloader,validloader,testloader,learning_rate,device,epochs):
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), learning_rate)
    epochs = epochs
    model.train(True)
    # change to device
    model.to(device)

    for e in range(epochs):
        steps = 0
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            valid_loss,valid_accuracy = valid(model,valiloader,device)
            test_accuracy = test(model,testloader,device)
            print("Epoch: {}/{}... ".format(e+1, epochs),
                     "Loss: {:.4f}".format(running_loss/steps))
            print("validation_loss: {:.4f}".format(valid_loss),"validation_accuracy: {:.4f}".format(valid_accuracy)) 
            #print( "Accuracy: {:.4f}".format(running_loss/steps))
            
    save_checkpoint(model,optimizer,epoch,train_datasets,filepath='new.pkl') 
    
    
def valid(model,validloader,device):
    correct = 0
    total = 0
    with torch.no_grad():
        model.train(False)
        model.to(device)
        running_loss = 0
        step=0
        for ii,(inputs,labels) in enumerate(validloader):
            step+=1
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    #print("Validation_Loss: {:.4f}".format(running_loss/step))
    return running_loss/step,(100 * correct / total) 

def test(model,testloader,device):
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        model.to(device)
        for ii,(inputs,labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
        
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return   (100 * correct / total)   

def save_checkpoint(model,optimizer,epoch,train_datasets,filename='new.pkl'):
    state = {
    'epoch': epoch,
    'state_dict': model.state_dict(),
    'optimizer':optimizer,
    'class_to_idx': train_datasets.class_to_idx,
            }
    torch.save(state, filename)
    
def load_checkpoint(model,filename='saved_model_3.pkl'):
    saved = torch.load(filename)
    #model=saved['state_dict']
    #model= models.resnet101(pretrained=True)
    #model.fc = saved['classifier']
    model.load_state_dict(saved['state_dict'])
    return model