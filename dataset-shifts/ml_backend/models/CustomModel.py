'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import abstractmethod 

from .encoder import celebA_Encoder
from .decoder import celebA_Decoder


class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()
        
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def get_latent(self, x):
        pass

    @abstractmethod
    def get_score(self, x):
        pass

    @abstractmethod
    def get_score_layer(self, x, layer):
        pass

    def get_all_layers(self):
        pass

    #@abstractmethod
    def get_final_layer(self):
        pass



class ComboModel(BaseModel):
    def __init__(self, cnn, drmodel):
        super(ComboModel, self).__init__()
        self.cnn = cnn
        self.drmodel = drmodel

        self.latent_size = drmodel.get_latent_size()

    def forward(self, x):
        x = self.cnn.get_latent(x) 
        ret = self.drmodel(x)
        self.z = self.drmodel.z
        return ret


    def get_latent(self, x):
        z_cnn = self.cnn.get_latent(x) 
        self.z = self.drmodel.get_latent(z_cnn)
        return self.z

    def get_layer(self, x, layer):
        return self.cnn.get_layer(x, layer)

    def get_all_layers(self):
        pass
        #return self.cnn.get_all_layers() + self.drmodel.get_all_layers()

    def get_final_layer(self):
        return self.drmodel.get_final_layer()

    def get_score(self, x):
        return self.forward(x)

    def get_score_layer(self, input_image, layer):
        _, out_layer = self.cnn.get_score_layer(input_image, layer)
        logit = self.drmodel.forward_logit(self.cnn.z)
        score = torch.sigmoid(-logit/5)


        return score, out_layer







class SugiyamaNet(BaseModel):
    def __init__(self, size):
        super(SugiyamaNet, self).__init__()
        self.fc1 = nn.Linear(size, size*2)
        self.fc2 = nn.Linear(size*2, 256)
        self.fc_out = nn.Linear(256, 1)

        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)

        self.leak = nn.LeakyReLU(0.2)

        self.latent_size = size

        self.all_layers = nn.ModuleList([self.fc1, self.leak,
                                         self.fc2])

    def forward(self, x):
        self.z = self.get_latent(x)
        x = F.softplus(torch.clamp(self.fc_out(self.z),min=-50, max=50))
        return x

    def forward_logit(self, x):
        self.z = self.get_latent(x)
        return self.fc_out(self.z)


    def get_latent(self, x):
        x = F.leaky_relu(self.dr1(self.fc1(x)), negative_slope=0.2)
        self.z = F.leaky_relu(self.dr2(self.fc2(x)), negative_slope=0.2)
        return self.z

    def get_layer(self, x, layer):
        pass

    def get_all_layers(self):
        pass

    def get_final_layer(self):
        return self.fc_out

    def get_score(self, x):
        return self.forward(x)

    def get_latent_size(self):
        return self.latent_size


class AutoEncoder(BaseModel):
    def __init__(self, latent_size = 128):
        super(AutoEncoder, self).__init__()
        self.enc = celebA_Encoder(latent_size)
        self.dec = celebA_Decoder(latent_size)
        self.latent_size = latent_size

    def forward(self, x):
        self.z = self.enc(x)
        return self.dec(self.z)

    def get_latent(self, x):
        return self.enc(x)


    def get_all_layers(self):
        pass

    def get_final_layer(self):
        return self.enc.latent_mapping[0]

    def get_latent_size(self):
        return self.latent_size

    def get_score(self, x):
        return torch.mean((x - self.forward(x))**2, (1,2,3))

    def get_score_layer(self, input_image, layer):
        x = input_image
        cur_layer = 0
        convlayers = [self.enc.layer0,self.enc.layer1,self.enc.layer2,self.enc.layer3,self.enc.layer4]
        for i, conv in enumerate(convlayers):
            x = conv(x)
            if i == layer: out_layer = x
        
        spatial_size = x.size(2)
        x = nn.functional.avg_pool2d(x, spatial_size, 1)
        x = x.view(x.size(0), -1)
        x = self.enc.latent_mapping(x)
        self.z = x
        
        score = torch.mean((input_image - self.dec(x))**2, (1,2,3))

        return score, out_layer
        




class VAE(nn.Module):
    def __init__(self, latent_size = 128):
        super(VAE, self).__init__()

        self.latent_size = latent_size
        self.enc = celebA_Encoder(latent_size)
        self.dec = celebA_Decoder(latent_size)

        self.fc1 = nn.Linear(latent_size, latent_size)
        self.fc2 = nn.Linear(latent_size, latent_size)

    def get_latent_size(self):
        return self.latent_size

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    

    def forward(self, x):
        enc_z = self.enc(x)
        mu, logvar = self.fc1(enc_z), self.fc2(enc_z)
        self.z = mu
        mu = mu.squeeze()
        logvar = logvar.squeeze()
        z = self.reparameterize(mu, logvar)
        return self.dec(z.view(-1, self.latent_size)), mu, logvar

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def get_latent(self, x):
        self.z = self.fc1(self.enc(x))
        return self.z

    def get_layer(self, x, layer):
        pass

    def get_all_layers(self):
        pass

    def get_final_layer(self):
        return self.fc1

    def get_score(self, x):
        mu = self.get_latent(x)
        mu = mu.squeeze()
        ret = torch.norm(mu, p=2, dim=1)
        if ret.shape[0] != x.shape[0]: exit("bad score for AutoEncoder")
        return ret

    def get_score_layer(self, input_image, layer):
        x = input_image
        cur_layer = 0
        convlayers = [self.layer0(x),self.layer1(x),self.layer2(x),self.layer3(x),self.layer4(x)]
        for i, conv in enumerate(convlayers):
            x = conv(x)
            if i == layer: out_layer = x
        
        spatial_size = x.size(2)
        x = nn.functional.avg_pool2d(x, spatial_size, 1)
        x = x.view(x.size(0), -1)
        x = self.latent_mapping(x)

        self.z = self.fc1(x)

        score = torch.norm(self.z.squeeze(), p=2, dim=1)
        
        return score, out_layer


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

    


class ResNet(BaseModel):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.latent_size = 32


        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear1 = nn.Linear(512*block.expansion*4, self.latent_size)
        self.linear2 = nn.Linear(self.latent_size, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        self.z = self.get_latent(x)
        return self.linear2(self.z)

    def get_latent(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        self.z = F.relu(out)
        return self.z

    def get_final_layer(self):
        return self.linear2

    def get_layer(self, x, layer):
        pass

    def get_score(self, x):
        return torch.sigmoid(self.forward(x))

    def get_latent_size(self):
        return self.latent_size

    def get_score_layer(self, input_image, layer):
        x = input_image

        x = F.relu(self.bn1(self.conv1(x)))
        convlayers = [self.layer0(x),self.layer1(x),self.layer2(x),self.layer3(x),self.layer4(x)]
        for i, conv in enumerate(convlayers):
            x = conv(x)
            if i == layer-1: out_layer = x
        
        out = F.avg_pool2d(x, 4)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        self.z = F.relu(out)
        
        score = torch.sigmoid(self.linear2(out))
        
        return score, out_layer


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def ResNet18(num_classes=1):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)

def vgg11(num_classes=1):
    return VGG(num_classes)

class VGG(nn.Module):
    def __init__(self, num_classes=1):
        super(VGG, self).__init__()
        self.features = self._make_layers([64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'])
        self.latent_size = 2048
        self.classifier = nn.Linear(2048, num_classes)
        self._initialize_weights()

    def forward(self, x):
        self.z = self.get_latent(x)
        return self.classifier(self.z)

    def get_latent(self, x):
        out = self.features(x)
        return out.view(out.size(0), -1)


    def get_final_layer(self):
        return self.classifier

    def get_layer(self, x, layer):
        pass

    def get_score(self, x):
        return F.softplus(self.forward(x))

    def get_latent_size(self):
        return self.latent_size

    def get_score_layer(self, input_image, layer):
        x = input_image

        for i,seq in enumerate(self.features):
            x = seq(x)
            if i == layer-1: out_layer = x
        
        score = F.softplus(self.classifier(x))
        
        return score, out_layer

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


