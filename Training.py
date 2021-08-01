import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from convnet import ConvNet

import sys
import time

#Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Tensorboard
#name = "MNIST/{}".format(int(time.time()))
name = "runs/mnist"

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/mnist")

#DATA & HYPERPARAMETERS
#Hyper parameters
num_epochs = 15
batch_size = 100
learning_rate = 0.001

#Dataset
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)


example_imgs, example_lbls = next(iter(train_loader)) #one batch
example_imgs = example_imgs.to(device)
example_lbls = example_lbls.to(device)

imgs_grid = torchvision.utils.make_grid(example_imgs) #display grid in tensorboard
writer.add_image('mnist_images',imgs_grid)

#getting the dimensions of one image, batch, len of dataset
print(f'dimension of one image {example_imgs[0].size()}')
print(f'shape of a batch {example_imgs.shape}')
print(f'length of dataset {len(train_dataset)}')

#MODEL
model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.5)

writer.add_graph(model,example_imgs.reshape(-1,1,28,28))

#TRAINING
n_total_steps = len(train_loader) 
print(f'total steps {n_total_steps}') #total_steps = total_length / batch_size

running_loss = 0.0
running_correct = 0

for epoch in range(num_epochs):
    for i, data in enumerate(train_loader): 
        images = data[0].to(device) #<images> dimension = batch*1*28*28 = 100*1*28*28
        labels = data[1].to(device)

        #Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        #Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch>10:
            scheduler.step()
                
        running_loss += loss.item()
        _,predicted = torch.max(outputs.data,1) #returns (value, index)
        running_correct += (predicted==labels).sum().item()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            writer.add_scalar("training loss",running_loss/100,epoch*n_total_steps+i) #adding graphs to tensorboard
            writer.add_scalar('accuracy',running_correct/100,epoch*n_total_steps+i)
            running_loss = 0.0
            running_correct = 0
        
print("finished training")

#VALIDATION 
with torch.no_grad():
    preds = [] #all probabilities (predictions) for test data
    labs = [] #all labels of test data
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.reshape(-1,1,28,28).to(device)
        labels = labels.to(device)
        outputs = model(images) #forward pass

        _,predictions = torch.max(outputs,1) #value, index

        n_samples += labels.shape[0]
        n_correct += (predictions==labels).sum().item()

        for i in range(batch_size): #looping through single images
            label = labels[i] 
            pred = predictions[i]
            if (label==pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1
            
        class_probabilities = [F.softmax(output, dim=0) for output in outputs] 
        preds.append(class_probabilities)
        labs.append(labels)
            
    preds = torch.cat([torch.stack(batch) for batch in preds]) #shape = 10000*10
    labs = torch.cat(labs) #shape = 10000
        
    classes = range(10)
    print(classes)
    for i in classes: #precision recall curve for each class
        labels_i = labs == i
        preds_i = preds[:,i]
        writer.add_pr_curve(str(i),labels_i,preds_i,global_step=0) 
        
    writer.close()

    acc= 100.0* n_correct / n_samples
    print(f'Accuracy of network: {acc} %')

#SAVING MODEL
FILE = "model.pth"
torch.save(model.state_dict(),FILE) #saving the parameters of the model


