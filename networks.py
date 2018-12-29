import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import io


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
#        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 5), nn.PReLU(),
#                                     nn.MaxPool2d(3, stride=2),
#                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
#                                     nn.MaxPool2d(3, stride=2))
        self.convnet = models.resnet18(pretrained=True)
        self.convnet_layer = self.convnet._modules.get('avgpool')
        self.convnet.eval()

        self.fc = nn.Sequential(nn.Linear(512, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 30)
                                )

    def forward(self, x):
        my_embedding = torch.cuda.FloatTensor(x.shape[0],512,1,1).fill_(0)

        def copy_data(m,i,o):
            my_embedding.copy_(o.data)

        h = self.convnet_layer.register_forward_hook(copy_data)

        self.convnet(x)
        h.remove()

        my_embedding = my_embedding.view(my_embedding.size()[0], -1)
        output = self.fc(my_embedding)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)

class textEmbedding(nn.Module):
    def __init__(self):
        super(textEmbedding, self).__init__()
        self.fc = nn.Sequential(nn.Linear(300, 128),
                nn.PReLU(),
                nn.Linear(128, 128),
                nn.PReLU(),
                nn.Linear(128, 30))

    def forward(self, x):
        return self.fc(x)


class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(2, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

class InterTripletNet(nn.Module):
    def __init__(self, image_embedding_net, text_embedding_net):
        super(InterTripletNet, self).__init__()
        self.image_embedding_net = image_embedding_net
        self.text_embedding_net = text_embedding_net

    def forward(self, a_v, p_t, n_t, a_t, p_v, n_v):
        output_av = self.image_embedding_net(a_v)
        output_pt = self.text_embedding_net(p_t)
        output_nt = self.text_embedding_net(n_t)
        output_at = self.text_embedding_net(a_t)
        output_pv = self.image_embedding_net(p_v)
        output_nv = self.image_embedding_net(n_v)

        return output_av, output_pt, output_nt, output_at, output_pv, output_nv

    def get_embedding(self, x):
        return self.image_embedding_net(x)

    def get_embedding_word(self, x):
        return self.text_embedding_net(x)
