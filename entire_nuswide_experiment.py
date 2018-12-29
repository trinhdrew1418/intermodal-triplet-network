import csv
from torchvision import transforms
from datasets_batched import NUS_WIDE
import torchvision as tv

import pickle

import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable

from trainer_batched import fit
import numpy as np

cuda = torch.cuda.is_available()

import matplotlib
import matplotlib.pyplot as plt

mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

print("Loading NUS_WIDE dataset...")
data_path = 'NUS_WIDE'
dataset = NUS_WIDE(root=data_path,
    transform=transforms.Compose([tv.transforms.Resize((224,224)), transforms.ToTensor(),
                                 transforms.Normalize(mean,std)]))
print("Done.")

# setting up labels
print("Loading in text labels...")
with open('labels.csv') as f:
    reader = csv.reader(f)
    NUS_WIDE_classes = [i[0] for i in list(reader)]
for i in range(len(NUS_WIDE_classes)):
    if '_' in NUS_WIDE_classes[i]:
        NUS_WIDE_classes[i] = NUS_WIDE_classes[i].split('_')[0]
    if NUS_WIDE_classes[i] == 'adobehouses':
        NUS_WIDE_classes[i] = 'adobe'
    if NUS_WIDE_classes[i] == 'kauai':
        NUS_WIDE_classes[i] = 'hawaii'
    if NUS_WIDE_classes[i] == 'oahu':
        NUS_WIDE_classes[i] = 'hawaii'
n_classes = len(NUS_WIDE_classes)
print("Done.")

# setting up dictionary
print("Loading in word vectors...")
text_dictionary = pickle.load(open("pickles/word_embeddings/word_embeddings_tensors.p", "rb"))
print("Done")
# setting up tag_matrix
print("Loading in tag matrix")
tag_matrix = pickle.load(open("pickles/nuswide_metadata/tag_matrix.p", "rb"))
print("Done")
# setting up concept_matrix
print("Loading in concept matrix")
concept_matrix = pickle.load(open("pickles/nuswide_metadata/concept_matrix.p", "rb"))
print("Done")


# creating indices for training data and validation data
print("Making training and validation indices...")
from torch.utils.data.sampler import SubsetRandomSampler

dataset_size = len(dataset)
validation_split = 0.3

indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
np.random.seed(21)
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
validation_sampler = SubsetRandomSampler(val_indices)
print("Done.")

# setting up loaders
from networks import textEmbedding
from losses import InterTripletLoss
from networks import InterTripletNet

batch_size = 256
kwargs = {'num_workers': 32, 'pin_memory': True} if cuda else {}
i_triplet_train_loader = torch.utils.data.DataLoader(dataset,  batch_size=batch_size, sampler=train_sampler, **kwargs)
i_triplet_val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=validation_sampler, **kwargs)

# Set up the network and training parameters
from networks import EmbeddingNet, TripletNet
from losses import TripletLoss

margin = 1.
text_embedding_net = textEmbedding()
image_embedding_net = EmbeddingNet()
model = InterTripletNet(image_embedding_net, text_embedding_net)
if cuda:
    model.cuda()
loss_fn = InterTripletLoss(1.0)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 100

fit(i_triplet_train_loader, i_triplet_val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, text_dictionary, NUS_WIDE_classes,
   tag_matrix, concept_matrix)

pickle.dump(model, open('entire_nuswide_model.p', 'wb'))
