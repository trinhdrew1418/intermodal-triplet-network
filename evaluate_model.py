import torch
import matplotlib.pyplot as plt
import pickle
import numpy as np
from datasets_batched import NUS_WIDE_KNN
from torchvision import transforms
import torchvision as tv
from torch.utils.data.sampler import SubsetRandomSampler
import csv
import faiss

model = pickle.load(open("pickles/models/entire_nuswide_model_2.p", "rb"))
relevancy_matrix = pickle.load(open("pickles/nuswide_metadata/relevancy_matrix.p", "rb"))

def main():
    train_loader, test_loader = make_loaders()
    t_indices, tag_rankings = make_tag_ranking(model, train_loader, test_loader, relevancy_matrix)
    im_to_ranked = make_image_ranked_relevancy_matrix(t_indices, tag_rankings)
    f1, precision, recall = f1_precision_recall(t_indices, im_to_ranked, k=7)

    print("mean F1: ", np.mean(f1))
    print("mean Precision: ", np.mean(precision))
    print("mean Recall: ", np.mean(recall))

def MiAP(im_and_tag_rankings):
    iAPs = np.zeros(len(test_loader.sampler))
    curr_idx = 0
    for im_idx, tag_ranking in im_and_tag_rankings:
        # R
        R_l = np.zeros(81)
        for i in range(len(R_l)):
            if relevancy_matrix[im_idx, i] == 1:
                R_l[i] = tag_ranking[i,0]
        R = np.sum(R_l)
        if R == 0:
            continue

        # summation
        S, rj  = 0, 0
        for j, tag in enumerate(tag_ranking):
            if relevancy_matrix[im_idx, tag[1]] == 1:
                rj += tag[0]
                S += np.divide(rj, j + 1)
        iAPs[curr_idx] = np.divide(1, R) * S
        curr_idx += 1
    return np.mean(iAPs)

def f1_precision_recall(t_indices, im_to_ranked_relevancy, k=3):
    top_k_relevant = im_to_ranked_relevancy[:,k]
    num_concepts = relevancy_matrix.shape[1]
    class_recalls = np.zeros(num_concepts)
    class_precisions = np.zeros(num_concepts)

    for concept_idx in range(num_concepts):
        relevant_images = np.nonzero(relevancy_matrix[:,concept_idx])[0]
        relevant_images = [idx for idx in relevant_images if idx < top_k_relevant.shape[0]]
        Ng_i = np.sum(relevancy_matrix[:, concept_idx])
        Nc_i = (concept_idx == top_k_relevant[relevant_images]).sum()
        Np_i = (concept_idx == top_k_relevant).sum()

        if Ng_i != 0:
            class_recalls[concept_idx] = Nc_i / Ng_i
        if Np_i != 0:
            class_precisions[concept_idx] = Nc_i / Np_i

    per_class_f1 = np.divide(np.multiply(2 * class_precisions, class_recalls), class_precisions + class_recalls)
    return per_class_f1, class_precisions, class_recalls

def make_loaders():
    NUS_WIDE_classes = []

   # init dataset
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    data_path = 'NUS_WIDE'

    dataset = NUS_WIDE_KNN(data_path,
        transforms.Compose([tv.transforms.Resize((224,224)), transforms.ToTensor(),
                                     transforms.Normalize(mean,std)]), NUS_WIDE_classes)
    # splitting up train and test:
    dataset_size = len(dataset)
    validation_split = 0.3

    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed(21)
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # init loaders
    batch_size = 256
    cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 8, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(dataset,  batch_size=batch_size, sampler=train_sampler, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, **kwargs)

    return train_loader, test_loader

"""
returns a tuple: idx of image in respects to relevancy matrix, ranking of the image
"""
def make_tag_ranking(model, train_loader, test_loader, relevancy_matrix):
    nearest_images, t_indices = faiss_similarity(model, train_loader, test_loader)

    # ranking[i,j,:] -> tuple corresponding to the jth relevant tag of image i
    # ranking[i,j,0]: count of tag
    # ranking[i,j,1]: column index in relevancy matrix
    ranking = np.zeros((nearest_images.shape[0], 81, 2), dtype=int)

    for i in range(nearest_images.shape[0]):
        for j in range(nearest_images.shape[1]):
            relevant_tags = np.argmax(relevancy_matrix[nearest_images[i,j],:])
            if type(relevant_tags) not in (list, tuple):
                relevant_tags = [relevant_tags]
            for k in relevant_tags:
                ranking[i,k,0] += 1
        ranking[i,:,1] += np.argsort(ranking[i,:,0])[::-1]
        ranking[i,:,0].sort()
        ranking[i,:,0] = np.flip(ranking[i,:,0], axis=0)

    return t_indices, ranking

def couple_indices_and_ranking(t_indices, ranking):
    return ((t_indices[i], ranking[i,:,:]) for i in range(ranking.shape[0]))

def make_image_ranked_relevancy_matrix(t_indices, ranking):
    largest_index = t_indices.max()
    image_to_ranked_relevancy = np.zeros((largest_index + 1, ranking.shape[1]))

    for img_idx, tag_ranking in zip(t_indices, ranking):
        for idx, count_concept in enumerate(tag_ranking):
            image_to_ranked_relevancy[img_idx, idx] = count_concept[1]
    return image_to_ranked_relevancy

"""
returns tuple:
- a matrix: item i,j: index of jth relevant image of image i
   (where image i is a query from the test loader)
   (where the index of jth relevant image is the index in the full dataset)
- a list: maps each index in the
"""
def faiss_similarity(model, train_loader, test_loader, k=30):
    base_db, b_indices = make_db(model, train_loader)
    test_db, t_indices = make_db(model, test_loader)

    index = faiss.IndexFlatL2(k)
    index.add(base_db)
    _, indices = index.search(test_db, k)

    for i in range(indices.shape[0]):
        for j in range(indices.shape[1]):
            indices[i,j] = b_indices[indices[i,j]] # convert query indices into indices in original dataset

    return indices, t_indices

"""
returns tuple:
- from embeddings produced by model, creates matrix for faiss training/querying
- returns a list: ith entry is an index in the original dataset that corresponds to the ith image in faiss_db
"""
def make_db(model, train_loader):
    cuda = True
    n = len(train_loader.sampler)
    d = 30 # size of the embeddings outputted by the model
    model.eval()

    faiss_db = np.empty((n,d), dtype='float32')
    fidx_to_idx = np.empty(n, dtype=int)

    n_idx = 0
    for _, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple,list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
        target = target.numpy()

        embeddings = model.get_embedding(*data)

        for idx in range(len(embeddings)):
            faiss_db[n_idx + idx, :] = embeddings[idx].cpu().detach().numpy()
            fidx_to_idx[n_idx + idx] = target[idx]
        n_idx += len(embeddings)

    return faiss_db, fidx_to_idx

main()

#print(MiAP(model,relevancy_matrix))
