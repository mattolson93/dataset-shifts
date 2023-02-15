'''Train CIFAR10 with PyTorch.'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data

from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import roc_curve, roc_auc_score

import numba

import os
import argparse
from tqdm import tqdm, trange


#import encoder
#from misc_functions import get_example_params, save_class_activation_images

import copy
from PIL import Image, ImageFilter
import matplotlib.cm as mpl_color_map

from torch.autograd import Variable


def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale

    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)

    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


def save_gradient_images(gradient, file_name):
    """
        Exports the original gradient image

    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    if not os.path.exists('results'):
        os.makedirs('results')
    # Normalize
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    # Save image
    path_to_file = os.path.join('results', file_name + '.jpg')
    save_image(gradient, path_to_file)


def save_class_activation_images(org_img, activation_map, file_name, save_original=True):
    """
        Saves cam activation map and activation map on the original image

    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'inferno')
    
    if save_original: save_image(org_img, file_name+'_original.png')
    save_image(heatmap, file_name+'_Cam_Heatmap.png')
    save_image(heatmap_on_image, file_name+'_Cam_On_Image.png')
    save_image(activation_map, file_name+'_Cam_Grayscale.png')


def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr


def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)


def preprocess_image(pil_im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): PIL Image or numpy array to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    #ensure or transform incoming image to PIL image
    if type(pil_im) != Image.Image:
        try:
            pil_im = Image.fromarray(pil_im)
        except Exception as e:
            print("could not transform PIL_img to a PIL Image object. Please check input.")

    # Resize image
    if resize_im:
        pil_im = pil_im.resize((224, 224), Image.ANTIALIAS)

    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im


def get_positive_negative_saliency(gradient):
    """
        Generates positive and negative saliency maps based on the gradient
    Args:
        gradient (numpy arr): Gradient of the operation to visualize

    returns:
        pos_saliency ( )
    """
    pos_saliency = (np.maximum(0, gradient) / gradient.max())
    neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
    return pos_saliency, neg_saliency




class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, dr_model, target_layer):
        self.model = model
        self.dr_model = dr_model
        self.target_layer = target_layer


    def forward_pass(self, x):
        if self.model._get_name() == "celebA_Encoder":
            return self.fp_encoder(x)
            
        m = self.model
        layer0 = F.relu(m.bn1(m.conv1(x)))
        layer1 = m.layer1(layer0)
        layer2 = m.layer2(layer1)
        layer3 = m.layer3(layer2)
        layer4 = m.layer4(layer3)
        out = F.avg_pool2d(layer4, 4)
        out = out.view(out.size(0), -1)
        out = m.linear1(out)
        z = F.relu(out)
        out = m.linear2(z)
        
        layer_dict = {
            0: layer0,
            1: layer1,
            2: layer2,
            3: layer3,
            4: layer4,
            5: z,
        }

        if self.dr_model is not None:
            out = self.dr_model(z) 

        return layer_dict[self.target_layer], out

    

eps = 1e-6

class ScoreCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.target_layer = target_layer

    @torch.no_grad()
    def _generate_cam_base(self, input_image):
        _, conv_output = self.model.get_score_layer(input_image, self.target_layer)
        
        # Get convolution outputs
        target = conv_output
        # Create empty numpy array for cam
        bs = target.shape[0]
        h = target.shape[2]
        w = target.shape[3]
        cam = torch.ones((bs, h, w)).cuda()
        # Multiply each weight with its conv output and then, sum
        for i in range(len(target[1])):
            # Unsqueeze to 4D
            saliency_map =  target[:, i, :, :].unsqueeze(1)
            # Upsampling to input size
            saliency_map = F.interpolate(saliency_map, size=(64, 64), mode='bilinear', align_corners=True)
            
            # Scale between [-1, 1]
            norm_saliency_map = self.normalize_cam(saliency_map, 64, neg_one_to_one=True)
            # Get the target score
            weight = self.model.get_score_layer(input_image*norm_saliency_map, self.target_layer)[0]
            cam += (weight.view(bs,1,1) * target[:, i, :, :])
        
        #remove boarder
        b = 1
        cam[:,:b,:]  = cam.min()
        cam[:,:,:b]  = cam.min()
        cam[:,-b:,:] = cam.min()
        cam[:,:,-b:] = cam.min()

        return cam.unsqueeze(1)
   
    def normalize_cam(self, AA, size, neg_one_to_one=False):
        eps = 1e-6
        bs = AA.shape[0]

        AA = AA.view(bs, -1)
        AA -= AA.min(1, keepdim=True)[0]
        AA /= AA.max(1, keepdim=True)[0] + eps
        ret = AA.view(bs, 1, size, size)

        if neg_one_to_one: 
            ret = 2*(ret - .5)

        return ret

    @torch.no_grad()
    def generate_cam(self, input_image, get_img=True):
        cam = self._generate_cam_base( input_image)
        cam = F.interpolate(cam, size=(64, 64), mode='bilinear', align_corners=True)
        cam = self.normalize_cam(cam, 64, neg_one_to_one=True).repeat(1, 3, 1, 1)
        cam = F.relu(cam)

        score = self.model.get_score(input_image * cam).cpu().numpy()
        return score, self.model.z

    @torch.no_grad()
    def save_cams(self, input_images, paths, outfile_base):
        bs = input_images.shape[0]
        cams = self._generate_cam_base( input_images)
        
        cams = F.interpolate(cams, size=(170, 170), mode='bilinear', align_corners=True)
        cams = self.normalize_cam(cams, 170, neg_one_to_one=False)
        cams = F.relu(cams)
        cams = cams.cpu().numpy()

        
        for i in range(input_images.shape[0]):
            cropped_img = transforms.CenterCrop(170)(Image.open(paths[i]))

            save_class_activation_images(cropped_img, cams[i].squeeze(), outfile_base+str(i))
        


    @torch.no_grad()
    def generate_cam_oneimage(self, input_image, get_img=True):
        # Full forward pass
        input_image = input_image.unsqueeze(0).cuda()
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        _, conv_output = self.model.get_score_layer(input_image, self.target_layer)
        
        # Get convolution outputs
        target = conv_output[0]
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i in range(len(target)):
            # Unsqueeze to 4D
            saliency_map = torch.unsqueeze(torch.unsqueeze(target[i, :, :],0),0)
            # Upsampling to input size
            saliency_map = F.interpolate(saliency_map, size=(64, 64), mode='bilinear', align_corners=False)
            if saliency_map.max() == saliency_map.min():
                continue
            # Scale between [-1, 1]
            norm_saliency_map = 2*(((saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())) - .5)
            # Get the target score
            w = self.model.get_score_layer(input_image*norm_saliency_map, self.target_layer)[0]
            cam += w.item() * target[i, :, :].detach().cpu().numpy()
        
        

        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        if not get_img: 
            cam = torch.from_numpy(2* (cam - .5)).cuda().unsqueeze(0).unsqueeze(0)
            cam = F.interpolate(cam, size=(64, 64), mode='bilinear', align_corners=False)
            score = self.model.get_score(input_image*cam).cpu().numpy()
            return score, self.model.z
        

        else:
            cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
            cam = np.uint8(Image.fromarray(cam).resize((170,
                           170), Image.ANTIALIAS))/255
            #cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
            #               input_image.shape[3]), Image.ANTIALIAS))/255
            return cam

def get_scorecam_example(net, img, path, outfile_base, dr_model=None, target_layer=1, invert_sigmoid=False):
    score_cam = ScoreCam(net, dr_model, target_layer=target_layer)
    cam = score_cam.generate_cam(img, invert_sigmoid)

    # Save mask
    save_class_activation_images(transforms.CenterCrop(170)(Image.open(path)), cam, outfile_base+str(target_layer))
