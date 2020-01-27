import sys
#sys.path.append('/opt/cocoapi/PythonAPI')
sys.path.append('\cocoapi\PythonAPI')
### to install COCO for windows use following command (ref. https://github.com/matterport/Mask_RCNN/issues/6)
### pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
### download test and traning data from http://cocodataset.org/#download
### download COCO API from https://github.com/cocodataset/cocoapi
from pycocotools.coco import COCO
#!pip install nltk
import nltk
nltk.download('punkt')
from data_loader import get_loader
from torchvision import transforms

# Define a transform to pre-process the training images.
transform_train = transforms.Compose([ 
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.RandomCrop(224),                      # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])

# Set the minimum word count threshold.
vocab_threshold = 5

# Specify the batch size.
batch_size = 10

# Obtain the data loader.
data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=True)


sample_caption = 'A person doing a trick on a rail while riding a skateboard.'

sample_tokens = nltk.tokenize.word_tokenize(str(sample_caption).lower())
print(sample_tokens)

sample_caption = []

#append start word
start_word = data_loader.dataset.vocab.start_word
print('Special start word:', start_word)
sample_caption.append(data_loader.dataset.vocab(start_word))
print(sample_caption)

##append all other words
sample_caption.extend([data_loader.dataset.vocab(token) for token in sample_tokens])
print(sample_caption)

### in order to convert a token to its corresponding integer, we call data_loader.dataset.vocab as a function. The details of how this call works can be explored in the __call__ method in the Vocabulary class in vocabulary.py.

##append end word
end_word = data_loader.dataset.vocab.end_word
print('Special end word:', end_word)

sample_caption.append(data_loader.dataset.vocab(end_word))
print(sample_caption)


## we convert the list of integers to a PyTorch tensor and cast it to long type. You can read more about the different types of PyTorch tensors on the website.
import torch

sample_caption = torch.Tensor(sample_caption).long()
print(sample_caption)


# Preview the word2idx dictionary.
dict(list(data_loader.dataset.vocab.word2idx.items())[:10])

# Print the total number of keys in the word2idx dictionary.
print('Total number of tokens in vocabulary:', len(data_loader.dataset.vocab))


"""
As you will see if you examine the code in **vocabulary.py**, the `word2idx` dictionary is created by looping over the captions in the training dataset.  If a token appears no less than `vocab_threshold` times in the training set, then it is added as a key to the dictionary and assigned a corresponding unique integer.  You will have the option later to amend the `vocab_threshold` argument when instantiating your data loader.  Note that in general, **smaller** values for `vocab_threshold` yield a **larger** number of tokens in the vocabulary.  You are encouraged to check this for yourself in the next code cell by decreasing the value of `vocab_threshold` before creating a new data loader. 
"""


"""
Step 2: Use the Data Loader to Obtain Batches

The captions in the dataset vary greatly in length. You can see this by examining data_loader.dataset.caption_lengths, a Python list with one entry for each training caption (where the value stores the length of the corresponding caption).

In the code cell below, we use this list to print the total number of captions in the training data with each length. As you will see below, the majority of captions have length 10. Likewise, very short and very long captions are quite rare. 
"""

from collections import Counter

# Tally the total number of training captions with each length.
counter = Counter(data_loader.dataset.caption_lengths)
lengths = sorted(counter.items(), key=lambda pair: pair[1], reverse=True)
for value, count in lengths:
    print('value: %2d --- count: %5d' % (value, count))


"""
To generate batches of training data, we begin by first sampling a caption length (where the probability that any length is drawn is proportional to the number of captions with that length in the dataset).  Then, we retrieve a batch of size `batch_size` of image-caption pairs, where all captions have the sampled length.  This approach for assembling batches matches the procedure in [this paper](https://arxiv.org/pdf/1502.03044.pdf) and has been shown to be computationally efficient without degrading performance.

Run the code cell below to generate a batch.  The `get_train_indices` method in the `CoCoDataset` class first samples a caption length, and then samples `batch_size` indices corresponding to training data points with captions of that length.  These indices are stored below in `indices`.

These indices are supplied to the data loader, which then is used to retrieve the corresponding data points.  The pre-processed images and captions in the batch are stored in `images` and `captions`.
"""

import numpy as np
import torch.utils.data as data

# Randomly sample a caption length, and sample indices with that length.
indices = data_loader.dataset.get_train_indices()
print('sampled indices:', indices)

# Create and assign a batch sampler to retrieve a batch with the sampled indices.
new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
data_loader.batch_sampler.sampler = new_sampler
    
# Obtain the batch.
images, captions = next(iter(data_loader))
    
print('images.shape:', images.shape)
print('captions.shape:', captions.shape)

# (Optional) Uncomment the lines of code below to print the pre-processed images and captions.
# print('images:', images)
# print('captions:', captions)

"""
Each time you run the code cell above, a different caption length is sampled, and a different batch of training data is returned. Run the code cell multiple times to check this out!

You will train your model in the next notebook in this sequence (2_Training.ipynb). This code for generating training batches will be provided to you.

    Before moving to the next notebook in the sequence (2_Training.ipynb), you are strongly encouraged to take the time to become very familiar with the code in data_loader.py and vocabulary.py. Step 1 and Step 2 of this notebook are designed to help facilitate a basic introduction and guide your understanding. However, our description is not exhaustive, and it is up to you (as part of the project) to learn how to best utilize these files to complete the project. You should NOT amend any of the code in either data_loader.py or vocabulary.py.

"""

"""
Step 3: Experiment with the CNN Encoder

Run the code cell below to import EncoderCNN and DecoderRNN from model.py. 
"""

# Watch for any changes in model.py, and re-load it automatically.
#% load_ext autoreload
#% autoreload 2

# Import EncoderCNN and DecoderRNN. 
from model import EncoderCNN, DecoderRNN

"""
In the next code cell we define a `device` that you will use move PyTorch tensors to GPU (if CUDA is available).  Run this code cell before continuing.
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (device)

"""
Run the code cell below to instantiate the CNN encoder in encoder.

The pre-processed images from the batch in Step 2 of this notebook are then passed through the encoder, and the output is stored in features.
"""

# Specify the dimensionality of the image embedding.
embed_size = 256

#-#-#-# Do NOT modify the code below this line. #-#-#-#

# Initialize the encoder. (Optional: Add additional arguments if necessary.)
encoder = EncoderCNN(embed_size)

# Move the encoder to GPU if CUDA is available.
encoder.to(device)
    
# Move last batch of images (from Step 2) to GPU if CUDA is available.   
images = images.to(device)

# Pass the images through the encoder.
features = encoder(images)

print('type(features):', type(features))
print('features.shape:', features.shape)

# Check that your encoder satisfies some requirements of the project! :D
assert type(features)==torch.Tensor, "Encoder output needs to be a PyTorch Tensor." 
assert (features.shape[0]==batch_size) & (features.shape[1]==embed_size), "The shape of the encoder output is incorrect."

"""
en flattened to a vector, before being passed through a Linear layer to transform the feature vector to have the same size as the word embedding.

Encoder

You are welcome (and encouraged) to amend the encoder in model.py, to experiment with other architectures. In particular, consider using a different pre-trained model architecture. You may also like to add batch normalization.

    You are not required to change anything about the encoder.

For this project, you must incorporate a pre-trained CNN into your encoder. Your EncoderCNN class must take embed_size as an input argument, which will also correspond to the dimensionality of the input to the RNN decoder that you will implement in Step 4. When you train your model in the next notebook in this sequence (2_Training.ipynb), you are welcome to tweak the value of embed_size.

If you decide to modify the EncoderCNN class, save model.py and re-execute the code cell above. If the code cell returns an assertion error, then please follow the instructions to modify your code before proceeding. The assert statements ensure that features is a PyTorch tensor with shape [batch_size, embed_size].
"""

"""
Step 4: Implement the RNN Decoder

Before executing the next code cell, you must write __init__ and forward methods in the DecoderRNN class in model.py.
 (Do not write the sample method yet - you will work with this method when you reach 3_Inference.ipynb.)

    The __init__ and forward methods in the DecoderRNN class are the only things that you need to modify as part of this notebook. 
    You will write more implementations in the notebooks that appear later in the sequence.

Your decoder will be an instance of the DecoderRNN class and must accept as input:

    the PyTorch tensor features containing the embedded image features (outputted in Step 3, when the last batch of images from Step 2 
    was passed through encoder), along with a PyTorch tensor corresponding to the last batch of captions (captions) from Step 2.

Note that the way we have written the data loader should simplify your code a bit. In particular, every training batch will contain pre-processed captions where all have the same length (captions.shape[1]), so you do not need to worry about padding.

    While you are encouraged to implement the decoder described in this paper, you are welcome to implement any architecture of your choosing, as long as it uses at least one RNN layer, with hidden dimension hidden_size.

Although you will test the decoder using the last batch that is currently stored in the notebook, your decoder should be written to accept an arbitrary batch (of embedded image features and pre-processed captions [where all captions have the same length]) as input. 
"""

# Specify the number of features in the hidden state of the RNN decoder.
hidden_size = 512

#-#-#-# Do NOT modify the code below this line. #-#-#-#

# Store the size of the vocabulary.
vocab_size = len(data_loader.dataset.vocab)

# Initialize the decoder.
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

# Move the decoder to GPU if CUDA is available.
decoder.to(device)
    
# Move last batch of captions (from Step 1) to GPU if CUDA is available 
captions = captions.to(device)

# Pass the encoder output and captions through the decoder.
outputs = decoder(features, captions)

print('type(outputs):', type(outputs))
print('outputs.shape:', outputs.shape)

# Check that your decoder satisfies some requirements of the project! :D
assert type(outputs)==torch.Tensor, "Decoder output needs to be a PyTorch Tensor."
assert (outputs.shape[0]==batch_size) & (outputs.shape[1]==captions.shape[1]) & (outputs.shape[2]==vocab_size), "The shape of the decoder output is incorrect."

