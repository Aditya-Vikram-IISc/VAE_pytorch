import torch
import torch.nn as nn
import torch.nn.functional as F



class VAE(nn.Module):
    def __init__(self, imgChannels = 1, featureDim = 32*20*20, zDim = 256):
        super(VAE, self).__init__()

        #Encoder
        self.En_Conv1 = nn.Conv2d(imgChannels, 16, 5)
        self.En_Conv2 = nn.Conv2d(16, 32, 5)
        self.En_fc1 = nn.Linear(featureDim, zDim)
        self.En_fc2 = nn.Linear(featureDim, zDim)

        #Decoder
        self.De_FC1 = nn.Linear(zDim, featureDim)
        self.Dec_ConvT1 = nn.ConvTranspose2d(32, 16, 5)
        self.Dec_ConvT2 = nn.ConvTranspose2d(16, imgChannels, 5)

    def encoder(self, x):

        x = F.relu(self.En_Conv1(x))
        x = F.relu(self.En_Conv2(x))
        x = x.view(-1, 32*20*20)

        mu = self.En_fc1(x)
        logVar = self.En_fc2(x)

        return mu, logVar

    def reparametrization(self, mu, logVar):

        std = torch.exp(logVar/2)
        eps = torch.rand_like(std)

        return mu + (eps*std)     

    def decoder(self, z):
        z = F.relu(self.De_FC1(z))
        z = z.view(-1, 32,20,20)
        z = F.relu(self.Dec_ConvT1(z))
        z = F.sigmoid(self.Dec_ConvT2(z))

        # z = torch.sigmoid(self.Dec_ConvT2(z))

        return z

    def forward(self, x:torch.Tensor):

        mu, logVar = self.encoder(x)
        z = self.reparametrization(mu, logVar)
        out = self.decoder(z)

        return out, z, mu, logVar