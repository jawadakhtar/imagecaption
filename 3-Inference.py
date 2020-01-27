###############Step 1: Get Data Loader for Test Dataset########

import torch
import torch.nn as nn
from torchvision import transforms
import sys
sys.path.append('/opt/cocoapi/PythonAPI')
from pycocotools.coco import COCO
from data_loader import get_loader
from torchvision import transforms

# TODO #1: Define a transform to pre-process the testing images.
transform_test = transforms.Compose([ 
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.RandomCrop(224),                      # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])

#-#-#-# Do NOT modify the code below this line. #-#-#-#

# Create the data loader.
data_loader = get_loader(transform=transform_test,    
                         mode='test')

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

# Obtain sample image before and after pre-processing.
orig_image, image = next(iter(data_loader))

# Visualize sample image, before pre-processing.
plt.imshow(np.squeeze(orig_image))
plt.title('example image')
plt.show()


######################Step 2: Load Trained Models#################################################


##checking cuda
print(torch.randn(2,2).cuda())
# if this works, you're in business

device = torch.device("cuda:0")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print ('CUDA detected!')
else:
    device = torch.device("cpu")
    print('CPU Detected!')


# Watch for any changes in model.py, and re-load it automatically.
#% load_ext autoreload
#% autoreload 2

import os
import torch
from model import EncoderCNN, DecoderRNN

# TODO #2: Specify the saved models to load.
encoder_file = 'encoder-3.pkl' 
decoder_file = 'decoder-3.pkl'

# TODO #3: Select appropriate values for the Python variables below.
embed_size = 300           # dimensionality of image and word embeddings
hidden_size = 512          # number of features in hidden state of the RNN decoder

# The size of the vocabulary.
vocab_size = len(data_loader.dataset.vocab)

# Initialize the encoder and decoder, and set each to inference mode.
encoder = EncoderCNN(embed_size)
encoder.eval()
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
decoder.eval()

# Load the trained weights.
encoder.load_state_dict(torch.load(os.path.join('./models', encoder_file)))
decoder.load_state_dict(torch.load(os.path.join('./models', decoder_file)))

# Move models to GPU if CUDA is available.
encoder.to(device)
decoder.to(device)

"""
Step 3: Finish the Sampler

Before executing the next code cell, you must write the sample method in the DecoderRNN class in model.py. This method should accept as input a PyTorch tensor features containing the embedded input features corresponding to a single image.

It should return as output a Python list output, indicating the predicted sentence. output[i] is a nonnegative integer that identifies the predicted i-th token in the sentence. The correspondence between integers and tokens can be explored by examining either data_loader.dataset.vocab.word2idx (or data_loader.dataset.vocab.idx2word).

After implementing the sample method, run the code cell below. If the cell returns an assertion error, then please follow the instructions to modify your code before proceeding. Do not modify the code in the cell below. 
"""


# Move image Pytorch Tensor to GPU if CUDA is available.
image = image.to(device)

# Obtain the embedded image features.
features = encoder(image).unsqueeze(1)

# Pass the embedded image features through the model to get a predicted caption.
output = decoder.sample(features)
print('example output:', output)

assert (type(output)==list), "Output needs to be a Python list" 
assert all([type(x)==int for x in output]), "Output should be a list of integers." 
assert all([x in data_loader.dataset.vocab.idx2word for x in output]), "Each entry in the output needs to correspond to an integer that indicates a token in the vocabulary."

# TODO #4: Complete the function.
""" It should take a list of integers (corresponding to the variable output in Step 3) as input and return the corresponding predicted sentence (as a single Python string). """
def clean_sentence(output):

    sentence = ""
    for i in output:
        #print(str(i) + "  " +  data_loader.dataset.vocab.idx2word[i])
        if (i != 0):
            sentence += " " + (data_loader.dataset.vocab.idx2word[i])
    return sentence

sentence = clean_sentence(output)
print('example sentence:', sentence)

assert type(sentence)==str, 'Sentence needs to be a Python string!'


"""Step 5: Generate Predictions!

In the code cell below, we have written a function (get_prediction) that you can use to use to loop over images in the test dataset and print your model's predicted caption."""

def get_prediction():
    orig_image, image = next(iter(data_loader))
    plt.imshow(np.squeeze(orig_image), cmap='gray')
    plt.title('Sample Image')
    plt.show()
    image = image.to(device)
    features = encoder(image).unsqueeze(1)
    output = decoder.sample(features)    
    sentence = clean_sentence(output)
    print(sentence)

get_prediction()