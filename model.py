import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

import torch.nn.functional as F

class DecoderRNN(nn.Module):
    
    ''' Initialize the layers of this model.'''
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size

        # embedding layer that turns words into a vector of a specified size
        self.embed = nn.Embedding(vocab_size, embed_size)

        # the LSTM takes embedded word vectors (of a specified size) as inputs 
        # and outputs hidden states of size hidden_dim
        self.lstm = nn.LSTM(embed_size, hidden_size)

        # the linear layer that maps the hidden state output dimension 
        # to the number of tags we want as output, tagset_size (in this case this is 3 tags)
        #self.hidden2tag = nn.Linear(hidden_size, tagset_size)
        self.linear = nn.Linear(hidden_size, vocab_size)

        # initialize the hidden state (see code below)
        self.hidden = self.init_hidden()


    def init_hidden(self):
        ''' At the start of training, we need to initialize a hidden state;
           there will be none because the hidden state is formed based on perviously seen data.
           So, this function defines a hidden state with all zeroes and of a specified size.'''
        # The axes dimensions are (n_layers, batch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size))

    
    # def forward(self, features, captions):
    #     #pass
    #     ''' Define the feedforward behavior of the model.'''
    #     # create embedded word vectors for each word in a captions
    #     embeds = self.word_embeddings(captions)
    #    
    #     # get the output and hidden state by passing the lstm over our captions
    #     # the lstm takes in our embeddings and hidden state
    #     lstm_out, self.hidden = self.lstm(
    #         embeds.view(len(captions), 1, -1), self.hidden)
    #    
    #     # get the scores for the most likely tag for a word
    #     tag_outputs = self.hidden2tag(lstm_out.view(len(captions), -1))
    #     tag_scores = F.log_softmax(tag_outputs, dim=1)
    #    
    #    
    #     tag_scores = F.log_softmax(lstm_out.view(len(captions), -1), dim=1)
    #    
    #     return tag_scores


    def forward(self, features, captions):
            captions = captions[:, :-1] #it return all captions, each having different lenghts 
            embedding = self.embed(captions)
            embedding = torch.cat((features.unsqueeze(dim = 1), embedding), dim = 1) #now adding dim to feathers and concat the embeddings (captions) to it
            lstm_out, hidden = self.lstm(embedding) #calling lstm
            outputs = self.linear(lstm_out) #make the output linear
            return outputs #return output


    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "       
        predictedSentence = []
        for index in range(max_len):
            lstm_out, states = self.lstm(inputs, states)
            
            lstm_out = lstm_out.squeeze(1)
            outputs = self.linear(lstm_out)
            predictedWord = outputs.max(1)[1]
            #print(predictedWord)
            predictedSentence.append(predictedWord.item())
            inputs = self.embed(predictedWord).unsqueeze(1)
        return predictedSentence
        