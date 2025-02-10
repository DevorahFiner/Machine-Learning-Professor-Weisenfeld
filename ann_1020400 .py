#!/usr/bin/env python
# coding: utf-8

# # Neural Network Homework
# 
# Your assignment is to build a neural network using fully connected (dense) layers and ReLU activations to classify images of fashion items. We will use subsets of the Fashion-MNIST dataset for training and testing our model. 
# 
# #### Save the Data
# Save the `train_dataset.pth` and `val_dataset.pth` in the same folder as this Jupyter Notebook. (If you use google colab, either upload it there, or save to your google drive and mount your google drive)
# 
# #### About the Data
# The data comes from the fashion-MNIST data set and consists of 28 X 28 pixel images of 10 types of fashion goods labeled as integers:
# 
#     0. T-shirt/top
#     1. Trouser
#     2. Pullover
#     3. Dress
#     4. Coat
#     5. Sandal
#     6. Shirt
#     7. Sneaker
#     8. Bag
#     9. Ankle boot
# 
# #### The Assignment
# Your job is to build a basic artificial neural network model to classify each image with its correct label. When you have fully trained your model, you should upload the **SAVED MODEL FILE** to BlackBoard.
# 
# #### Submitting the Assignment
# It should be named "ann_college_id_num.pth" (replace 'college_id_num' with your college id number).
# 
# For example, if your college id is 1234567, you should name your file ann_1234567.pth.
# 
# This is different than earlier assignments. You will not upload a .py file, you will upload a **.pth file**. You are not uploading code, but rather a trained Neural Network Model.
# 
# You may optionally upload your completed jupyter notebook file as well, but I will only check that if something goes wrong with your model file for partial credit.
# 
# #### Assignment Grading
# Yor uploaded model file will be tested on a withheld dataset (one that you don't have access to) I have included a validation dataset for you to test your model on here.
# 
# You will get a perfect score on this assignment if your model achieves an F1-score >- 0.85 on my test set. 
# 
# Your assignment grade will be computed as follows: $\text{Assignment Grade} = 100 \times \frac{\text{F1_score} - 0.1}{0.75} $
# 
# The F-1 score you get on the included validation set will likely be very close to the one you get on my test set. (my model achieved > 0.86 F-1 score on the included validation set and > 86% accuracy)
# 

# ## Load libraries
# *Note you may need to install some of these*

# In[30]:


pip install torchvision


# In[31]:


pip install torch


# In[32]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import numpy as np
from IPython.display import display, clear_output
from contextlib import nullcontext


# ## Load the Data

# In[33]:


#  Code to load datasets They should be saved to the same folder as this Jupyter Notebook.
#  If you are using Google Colab, you can upload them to the file system in colab 
#  or you can save them to your google drive and mount your drive

# Upload the datasets from your file system
train_dataset = torch.load('train_dataset.pth')
val_dataset = torch.load('val_dataset.pth')


# Create data loaders
batch_size = 64 # If you have memory issues, consider reducing the batch_size

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

label_map = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}


# ## Explore the Data

# #### Number of Samples

# In[34]:


print(f"training samples:   {len(train_dataset)}")
print(f"validation samples: {len(val_dataset)}")


# #### Check out a sample

# In[35]:


dataset = train_dataset  # feel free to switch to val_dataset to check that out
i = 0  # feel free to change to see any other samples

print(f"sample data type: {type(dataset[i])}")
print(f"tuple size: {len(dataset[i])}")
print(f"types of tuple members: {[type(j) for j in dataset[i]]}")


# Each sample in the datasets is a tuple with two elements. The first element is torch.Tensor object representing the pixels of a grayscale image, and the second member is an integer value representing the class label.

# #### Check out labels

# In[36]:


labels = np.array([lbl for _, lbl in dataset])
unique_labels, counts = np.unique(labels, return_counts=True)
for lbl, cnt in zip(unique_labels, counts):
    print(f"label: {lbl}, count: {cnt} - ({label_map[lbl]})")


# #### Check out X

# In[10]:


print(f"dimensions: {dataset[i][0].size()}")
print(f"value range: {[dataset[i][0].min().item(), dataset[i][0].max().item()]}")


# Images are usually stored as 3 dimensional tensor. 
# 
# The first dimension represents the color channels. Because this is a grayscale image, there is only one channel. Color images usually have 3 or 4 channels (one for each primary color red, green, and blue, and optionally one more representing transparency).
# 
# The second and third dimensions represent the height and width, respectfully of the image in pixels. In this case you can see that each image is 28 X 28 pixels.
# 
# The value of each pixel represents its intensity. This is usually a value between 0 no intensity (black) and 255 full pixel intensity (white if grayscale), but in our case it has been normalized to range from -1 (black) to 1 (white), because those values work better with neural networks.

# In[11]:


dataset[i][0]


# #### View images

# In[12]:


def imshow(img, negate=False): # This plots an image using matplot lib
    img = img / 2 + 0.5  # unnormalize
    img = -img if negate else img
    npimg = img.numpy()
    plt.figure(figsize=(2, 2))
    plt.imshow(npimg[0], cmap='gray')
    plt.xticks([])  # remove x tick marks
    plt.yticks([])  # remove y tick marks
    plt.show()

def datadisplay(sample, negate=False): # Shows the image with its label
    img, label = sample
    print(label_map[label])
    imshow(img, negate)


# In[17]:


datadisplay(dataset[i], True)


# ## Define the neural network model

# In[18]:


# This is the main part of your assignment.

#Code an Artificial Neural Network Class
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        
        # Add at least two hidden layers, and remember to use non-linear activations ReLUs between each layer. 
        # Your first hidden layer, must take an each image as a vector
        # currently, it is a 28 X 28 matrix...what is the input dim that you need when it's converted to a vector?
        # You get to decide the output_dim for the first hidden layer and the input and output dims for the rest of them.
        # Experiment with different numbers of layers and dimensions
        
        # You get to decide the input dimensions of the output layer, what output dim is required?
        
        # Add your nonlinearities as well, You can add them here or in the forward method
        
        self.layer1 = nn.Linear(28 * 28, 128)  # Input layer to first hidden layer
        self.layer2 = nn.Linear(128, 64)       # First hidden layer to second hidden layer
        self.layer3 = nn.Linear(64, 10)        # Second hidden layer to output layer
        self.relu = nn.ReLU()        
        
    def forward(self, X):
        # This is the Forward Propagation method that sends X through your model to get the model outputs
        
        # X will come in as 4 dimensional tensor.
        # The first dimension is the batch dimension = batch_size, 
        # the second is the channel dimension = 1
        # The third and fourth are the height and width, each = 28
        
        # You need to convert X into a two dimensional tensor: with batch_size rows and as input_dim columns
        
        # next you need to pass X through all your layers in succession, 
        # don't forget to apply nonlinear activations
        
        # This method should return a set of probability predictions. The shape should be batch_size by classes
        X = X.view(X.size(0), -1)  # Flatten the input
        X = self.relu(self.layer1(X))
        X = self.relu(self.layer2(X))
        X = self.layer3(X)
        return X


# #### Quick Model Test

# In[19]:


# The following code takes stacks the first two samples as a mini batch and runs them through the model
quick_batch = torch.stack((train_dataset[0][0],train_dataset[1][0]))

# If your model is coded correctly, you should get a two dimensional Tensor output of size = [2, output_dim]
quick_output = ANN().forward(quick_batch)
print(quick_output)
quick_output.size()


# ## Define the training loop

# In[20]:


# This function runs and manages a forward pass for an entire epoch. Feel free to study it.
def forward_pass(model, data_loader, loss_fn=None, optimizer=None, train=True, device=torch.device("cpu")):
    if train:
        model.train() # this allows the model to collect gradients
    else:
        model.eval() # this turns off gradient collection
    
    # Initialize variables
    loss, correct, total = 0.0, 0, 0 
    losses, accuracies, f1s = [], [], []
    preds, labels = [], []
    
    with torch.no_grad() if not train else nullcontext(): # little hack to use the same loop for both
        for imgs, lbls in data_loader: # loops through each batch
            imgs, lbls = imgs.to(device), lbls.to(device) # loads it on device
            outputs = model(imgs) # same thing is model.forward(imgs) built in to nn.Module super class
            
            batch_loss = loss_fn(outputs, lbls) if loss_fn else torch.tensor([0]) # dummy for loss if missing
            if train: # then we gotta do backward propagation too
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
            # gather batch stats
            loss += batch_loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == lbls).sum().item()
            total += len(predicted)
            preds.extend(predicted.cpu().numpy())
            labels.extend(lbls.cpu().numpy())
    
    # Compute and return epoch stats
    loss /= len(data_loader)
    accuracy = correct / total
    f1 = f1_score(labels, preds, average='weighted')
    return model, preds, labels, loss, accuracy, f1


# In[21]:


# This function plots metrics the way I want to.
def plot_metrics(fig, ax, train_metric, val_metric, epochs=None, metric='Loss'):
    ax.clear()
    epochs = len(train_metric) if not epochs else epochs
    ax.set_xlim(1, epochs)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Epochs')
    ax.set_ylabel(metric)
    ax.set_title('Training & Validation '+ metric)
    ax.plot(range(1, len(train_metric) + 1), train_metric, label='Training' + metric)
    ax.plot(range(1, len(val_metric) + 1), val_metric, label='Validation ' + metric)
    ax.legend()
    display(fig)
    plt.close(fig)
    return fig, ax


# In[22]:


# This is the training loop! I suggest training your model for no more than 25 epochs

def train_model(model, loss_fn, optimizer, train_loader, val_loader, epochs=10):
    # If you have a GPU, and installed cuda this will use it, but it isn't necessary
    # Google Colab allows you to access a GPU for free, but you need to select it up top
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    print(f"device: {'gpu' if device == torch.device('cuda') else device}")
    
    # Lists to collect stuff
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_f1s, val_f1s = [], []
    
    plt.ioff() #  turns off interactive plotting so to help with dynamically updating charts
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for epoch in range(epochs):
        
        # Forward Pass (train)
        _, _, _, train_loss, train_acc, train_f1 = forward_pass(model, train_loader, loss_fn, optimizer, 
                                                                train=True, device=device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        train_f1s.append(train_f1)
        
        # Forward Pass (validation)
        _, _, _, val_loss, val_acc, val_f1 = forward_pass(model, val_loader, loss_fn, train=False, device=device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_f1s.append(val_f1)        
        
        
        # Update the plot and print stats
        clear_output(wait=True) # Clear the output window
        plot_metrics(fig, ax, train_losses, val_losses, epochs)
        
        # Since we cleared the output window, we loop through the previous epochs' stats and reprint 
        for epoch, (train_loss, val_loss, train_acc, val_acc, train_f1, val_f1) in enumerate(
        zip(train_losses, val_losses, train_accuracies, val_accuracies, train_f1s, val_f1s)
        ):
            print(f'Epoch {epoch+1:2d}/{epochs:2d}, '
                  f'Train Loss: {train_losses[epoch]:.4f}, Val Loss: {val_losses[epoch]:.4f}, ' 
                  f'Train Acc: {train_accuracies[epoch]:.4f}, Val Acc: {val_accuracies[epoch]:.4f}, ' 
                  f'Train F1: {train_f1s[epoch]:.4f}, Val F1: {val_f1s[epoch]:.4f}')
    
    plt.ion() #  turn interactive plotting back on
    
    # Return final model results
    return train_losses, val_losses, train_accuracies, val_accuracies, train_f1s, val_f1s

# Train the model

# Initalize the model, loss_fn, and optimizer
model = ANN()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train it!
train_losses, val_losses, train_accuracies, val_accuracies, train_f1s, val_f1s = train_model(
    model, loss_fn, optimizer, train_loader, val_loader, epochs=25
)


# ## Plot the training and validation metrics

# In[23]:


plt.ioff()
    
fig, ax = plt.subplots(figsize=(10, 5))
plot_metrics(fig, ax, train_losses, val_losses,metric='Loss')

fig, ax = plt.subplots(figsize=(10, 5))
plot_metrics(fig, ax, train_accuracies, val_accuracies, metric='Accuracy')

fig, ax = plt.subplots(figsize=(10, 5))
plot_metrics(fig, ax, train_f1s, val_f1s, metric='F1-Score');


# ## Test the trained model

# In[28]:


# Function to save the model
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    print(f'Model loaded from {path}')
    return model


# In[29]:


# Save the model
file_name = 'ann_1020400.pth' #CHANGE THIS TO "ann_college_id_num.pth" (replace 'college_id_num' with your college id number)
save_model(model, file_name) 

# Create a new instance of the model and load the saved state
# changing the variable name to loaded_model so you can confirm it works
loaded_model = ANN() 
loaded_model = load_model(model, file_name)


# In[26]:


# Function to plot confusion matrix
def plot_confusion_matrix(labels, preds, classes):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
def test_model(model, data_loader):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"device: {'gpu' if device == torch.device('cuda') else device}")
    
    model = model.to(device)
    
    model, preds, labels, loss, accuracy, f1 = forward_pass(model, data_loader, train=False, device=device)
    f1_per_class = f1_score(labels, preds, average=None)
    
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test F1 Score: {f1:.4f}')
    for i, f1_class in enumerate(f1_per_class):
        print(f"Class {i} ({label_map[i] + ')':<12} F1 Score: {f1_class:.4f}")
    
    plot_confusion_matrix(labels, preds, label_map.values())
    return accuracy, f1, f1_per_class


# In[27]:


loaded_model = ANN()
loaded_model = load_model(model, file_name)
# Test the model
val_accuracy, val_f1, val_f1_per_class = test_model(loaded_model, val_loader)

