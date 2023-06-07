'''
Author: ccsmm@etri.re.kr
Usage : Copy following two lines to your code and remove remark at the beginning of the line to choose the mode among "Interactive" and "Agg" :
        Mode of tim.init() in main.py will affect all sub-module where import tim. So make sure mode of main.py and do not call tim.init() in sub-modules
#plt_mode="Interactive"  # Default, interactive mode in remote terminal with imshow, plot, etc. and save image also.
plt_mode="Agg"          # mode only to save figure to image in server without any of imshow, plot. It does not need remote terminal.
import sys;sys.path.insert(0,'/home/ccsmm/dg_git/ccsmmutils');import torch_img_utils as tim;tim.init(plt_mode)
'''

import cv2
import numpy as np
#import matplotlib.pyplot as plt  # use init(mode) instead of it.
from PIL import Image
import os
import torch
from torchvision.transforms.functional import normalize
import torchvision
import skimage
import skimage.io
import skimage.transform

from ipdb import set_trace as bp

global plt
global plt_mode
plt_mode = "Interactive"

def plt_init(mode=None, verbose=False):
    init(mode=None, verbose=False)

def init(mode=None, verbose=False):
    global plt
    global plt_mode
    if mode is None:
        mode = plt_mode
    if mode.lower() == 'agg':
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        if verbose:
            print("Matplolib : Agg mode in /home/ccsmm/dg_git/ccsmmutils/torch_img_utils")
    else:  # Interactive mode
        import matplotlib.pyplot as plt
        if verbose:
            print("Matplolib : Interactive mode in /home/ccsmm/dg_git/ccsmmutils/torch_img_utils")
    plt_mode = mode  # update with user input

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)

def show_cam_on_image(img, mask, ofname=None, colormap=cv2.COLORMAP_JET):  # img is floating type data, 0 ~1.0
    """ 
        img = cv2.imread(fname)
        img = np.float32(img_cv2) / 255
    """
    if type(img) == str:
        img = cv2.imread(img)
        img = np.float32(img) / 255
    try:
        if img.dtype is not np.dtype('uint8'):
            img = np.float32(img)  # To do : / img.max() after checking maximum

        h,w = mask.shape[-2], mask.shape[-1]
        cv2.resize(img, (h,w), interpolation = cv2.INTER_CUBIC)

        heatmap_mask = cv2.applyColorMap(np.uint8(255 * mask), colormap)
        heatmap_mask = np.float32(heatmap_mask) / 255 
        cam = heatmap_mask + np.float32(img)
        cam = cam / np.max(cam)
        img_jet = np.uint8(255 * cam)
        #img_jet = deprocess_image(img_jet)
        if ofname != None:
            cv2.imwrite(ofname, img_jet)
    except:
        bp()
        img_jet = img
    return img_jet

def cv_to_tensor(image, device="cpu"):  # "cpu" or "cuda" for device
    #image = np.random.rand(480,640,3)
    #if type(image) is np.ndarray:
    if type(image) is type(np.random.rand(480,640,3)):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # cv to PIL
        tensor_img = torch.Tensor(np.transpose(image,(2,0,1))).to(device)  # h,w,c ==> c,h,w
        return tensor_img
    else:
        return image

def tensor_to_cv(image):
    # image = np.random.rand(3, 480,640)
    # image = torch.Tensor(image).to('cuda')
    image = image.cpu().numpy()
    # a_mask = np.array(mask.cpu(), dtype=np.uint8)  # [ch, h, w]
    cv_img = np.transpose(image,(1,2,0))  # c,h,w ==> h,w,c : 480*640*3
    if image.shape[0] == 3:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)  # PIL to cv
    return cv_img

def img_cv2_to_plt(imgBGR):  # BGR to RGB
    return cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)

def img_plt_to_cv2(imgRGB):  # RGB to BGR
    return cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)

def ImgFileResize(fname, h=1000, w=1000):
    img = Image.open(fname)
    img = img.resize((w, h), Image.ANTIALIAS)
    img.save(fname)

def image_crop( infilename , save_path):
    """
    image file 와 crop한이미지를 저장할 path 을 입력받아 crop_img를 저장한다.
    :param infilename:
        crop할 대상 image file 입력으로 넣는다.
    :param save_path:
        crop_image file의 저장 경로를 넣는다.
    :return:
    """
 
    img = Image.open( infilename )
    (img_h, img_w) = img.size
    print(img.size)
 
    # crop 할 사이즈 : grid_w, grid_h
    grid_w = 96 # crop width
    grid_h = 96 # crop height
    range_w = (int)(img_w/grid_w)
    range_h = (int)(img_h/grid_h)
    print(range_w, range_h)
 
    i = 0
 
    for w in range(range_w):
        for h in range(range_h):
            bbox = (h*grid_h, w*grid_w, (h+1)*(grid_h), (w+1)*(grid_w))
            print(h*grid_h, w*grid_w, (h+1)*(grid_h), (w+1)*(grid_w))
            # 가로 세로 시작, 가로 세로 끝
            crop_img = img.crop(bbox)
 
            fname = "{}.jpg".format("{0:05d}".format(i))
            savename = save_path + fname
            crop_img.save(savename)
            print('save file ' + savename + '....')
            i += 1
 
def cv2_imshow_images(imglist, title='new', imgnumlist=[],
        color=(0, 255, 0), sz=(160,120), axis=0, dispEn=True):
    imgstack = []
    (h , w) =  sz
    if type(imglist) == type(list()):
        for idx, fname in enumerate(imglist):
            img = cv2.imread(fname)
            img = cv2.resize(img, (w,h))
            if len(imgnumlist) == len(imglist):
                atext = "{}".format(imgnumlist[idx])
            else:
                atext = "{}".format(idx)
            cv2.putText(img, atext, (int(w/2), 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)  # BGR
            if(len(imgstack) == 0):
                imgstack = img
            else:
                imgstack = np.hstack((imgstack, img))
    else:  # type(imglist) == type(str())
        img = cv2.imread(imglist)
        imgstack = cv2.resize(img, (160,120))

    if dispEn:
        cv2.imshow(title, imgstack)
        cv2.waitKey(1)
    return imgstack

def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    imga=imgreshape(imga)
    imgb=imgreshape(imgb)

    ha,wa = imga.shape[:2]
    hb,wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa+wb
    new_img = np.zeros(shape=(max_height, total_width, 3))
    new_img[:ha,:wa]=imga
    new_img[:hb,wa:wa+wb]=imgb
    return new_img

def concat_n_images(image_path_list):
    """
    Combines N color images from a list of image paths.
    """
    output = None
    for i, img_path in enumerate(image_path_list):
        img = plt.imread(img_path)[:,:,:3]
        if i==0:
            output = img
        else:
            output = concat_images(output, img)
    return output

def tonumpy(data):
    try: data.type()
    except :
        return data
#    if ( ('torch' in type(data)) and ('Tensor' in type(data)) ):
#        if ('cuda' in type(data)):
    if ( ('torch' in data.type()) and ('Tensor' in data.type()) ):
        if ('cuda' in data.type()):
            data=data.detach().cpu().numpy()
        else:
            data=data.detach().numpy()
    return data


def suptitle(title="No title"): # main title
    plt.suptitle(title)

def title(title="No title"): # subplot title
    plt.title(title)

def plot(x=None, y=None, sp=111, title="No title",t=0.000001,fdir='./',save=0, color=None, dispEn=False):
    if y == None:
        y = x
        x = np.arange(len(y))

    if x == None:
        x = np.arange(len(y))

    y = tonumpy(y)

    if sp != 0:  # If you want draw graph at the sampe sp as previous, then set sp to zero, otherwise you will meet warning.
        subplot(sp)

    if color is None:
        plt.plot(x, y)
    else:
        plt.plot(x, y, color)  # example of color : "ro-", "g^.", "bd:"

    plt.title(title)

    if dispEn:
        if(t>0):
            plt.draw()
            plt.pause(t)
        else:
            plt.show()

    if save:
        fname=os.path.join(fdir,title+'.png')
        plt.savefig(fname,bbox_inches='tight')

def plotxy(x, y, sp=111,title="No title",t=0.000001,fdir='./',save=0, dispEn=True):
    if len(np.shape(y)) != 1:
        print("dimension not matched, getting last element")
        y = y[-1]

    data = tonumpy(y)

    subplot(sp)
    plt.plot(x, data)
    plt.title(title)

    if dispEn:
        if(t>0):
            plt.draw()
            plt.pause(t)
        else:
            plt.show()

    if save:
        fname=os.path.join(fdir,title+'.png')
        plt.savefig(fname,bbox_inches='tight')


def visualize_feature(features, square=8, batch_idx=0, cmap='gray',
        dispEn=True, saveEn=False, fname='noname.png', image=None):  # output of NN, [batch, depth, h, w], ex) [4, 512, 30, 40]
    # visualize 64 features from each layer (although there are more feature maps in the upper layers)
    # From https://debuggercafe.com/visualizing-filters-and-feature-maps-in-convolutional-neural-networks-using-pytorch/
    feat = features[batch_idx]
    ch = len(feat)
    square = min(int(ch/2), square)
    ch_idx = 0
    plt.figure(figsize=(15, 15))
    for _y in range(square):
        for _x in range(square):
            sp = (square, square, ch_idx+1)
            feat = features[batch_idx, ch_idx, :, :]  # shape is [batch, channel, H ,W]
            imshow(feat, sp=sp, title='', dispEn=dispEn, cmap=cmap)
            if _y == 0 and _x == 0:
                if image is not None:
                    imshow(image, sp=sp, dispEn=dispEn, title='')
            ch_idx += 1
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close()
        
def subplot(sp):
    if type(sp) is tuple: #(8,8,10), When last digit is larger than 9.
        if len(sp) == 3:
            plt.subplot(sp[0], sp[1], sp[2])
    elif type(sp) is int:  #(889), When last digit is smaller than 10.
        plt.subplot(sp)

def imshow(img, sp=111, title="No title", t=0.000001,
        saveEn=False, dispEn = True, alpha=1.0, cmap='viridis', fname='noname.png',
        axis='off'):
    if len(np.shape(img)) == 4:   #batch_size * rgb
        img=img[-1] #Take last image(3,480,640)
    elif len(np.shape(img)) == 3: #rgb
        img=img
    elif len(np.shape(img)) == 2: #gray
        img=img
    else:
        print("dimension not matched")
        return 0

    if isinstance(img, torch.Tensor):
        img=img.to('cpu')
        img=tonumpy(img)
        img=imgreshape(img)


    subplot(sp)  # The type of sp should be one of the int or tuple : 443 or (4,4,10)
    plt.axis(axis)
    img = img.squeeze()
    plt.imshow(img, alpha=alpha, cmap=cmap)
    plt.title(title)
    
    if dispEn :
        if(t>0):
            plt.draw()
            plt.pause(t)
        else:
            plt.show()

    if saveEn:
        plt.savefig(fname, bbox_inches='tight', dpi=300)

def fig(num=0, figsize=(15, 15)):
    if num:
        plt.figure(num, figsize=figsize)
    else:
        plt.figure(figsize=figsize)

def cla():
    plt.cla()

def clf():
    plt.clf()

def imsshow(imgs, sp=111, title="No title", t=0.000001,
        saveEn=False, dispEn = True, alpha=1.0, cmap='viridis', fname='noname.png',
        axis='off'):
    img = torchvision.utils.make_grid(imgs)
    imshow(img, sp=sp, title=title, t=t,
        saveEn=saveEn, dispEn=dispEn, alpha=alpha, cmap=cmap, fname=fname,
        axis=axis)

def imgreshape(img):
    """img must be a tensor type data"""
    if isinstance(img,torch.Tensor):
        img=img.squeeze()
    if len(img.shape) == 2:
#        img=img.unsqueeze(0)
        np.reshape(img,[1,np.shape(img)[0],np.shape(img)[1]])

    if len(img.shape) == 3:
        if img.shape[0] <= 3:
            img=np.transpose(img,(1,2,0))

    return img

def imgnormalize(img):
    return cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)

#Default color space in OpenCV is BGR, but matplotlib's is RGB.
#So when we use matplotlib to disply it, we need to change the color space from bgr to rgb

def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def rgb2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def test_dispgen(model,imgL,imgR,idx):
        model.eval()
#        imgL = torch.FloatTensor(imgL).cuda()
#        imgR = torch.FloatTensor(imgR).cuda()    
        imgL = imresize(imgL,(368,1232))
        imgR = imresize(imgR,(368,1232))

        imgL=imgL.astype(np.float32)/256
        imgR=imgR.astype(np.float32)/256

        imgL = imgL.transpose(2,0,1)
        imgR = imgR.transpose(2,0,1)

        imgL=imgL.reshape(1,3,368,1232)
        imgR=imgR.reshape(1,3,368,1232)

        imgL = Variable(torch.FloatTensor(imgL).cuda())
        imgR = Variable(torch.FloatTensor(imgR).cuda())

        with torch.no_grad():
            output = model(imgL,imgR)
        output = torch.squeeze(output)
        pred_disp = output.data.cpu().numpy()
        display_save(imgL,imgR,pred_disp,idx)

        return pred_disp

def display_save(imgL,imgR,dispL,inx):
#    output_dir='output_eval_disp'
    output_dir=argsglb.output_dir
    plt.clf()
    imshow(imgL,sp=221,title='Left')
    imshow(imgR,222,'Right')
    imshow(dispL.astype('uint16'),223,'Disp(est)')
#   mshow(dispL_GT.astype('uint16'),224,'Disp(GT)')
    fname=os.path.join(output_dir,'psmnet_all_{}'.format(inx))
    plt.savefig(fname,bbox_inches='tight')
    fname=os.path.join(output_dir,'psmnet_disp_{}.png'.format(inx))
#    skimage.io.imsave(fname,(dispL.squeeze()*256).astype('uint16'))
    return 0

def heatmap(imgname, feature, ofname=None, brightness=0.0, contrast=1.0, normEn=True, dispEn=False):
    '''
    imgname can be image path or cv2 image(np.ndarray)
    '''
    return my_attention_map(imgname, feature, ofname, brightness=brightness, contrast=contrast, normEn=normEn, dispEn=dispEn)

def my_attention_map(imgname, feature, ofname=None, brightness=0.0, contrast=1.0, normEn=True, dispEn=False):
    GAP = torch.nn.AdaptiveAvgPool2d((1, 1))  # 512, H, W ==> 512, 1, 1
    
    feature_2d_norm = feature.mean(dim=0)  # Average in channel direction. [512, H, W] ==> [H, W]
    feature_2d_norm = feature_2d_norm - feature_2d_norm.min()
    feature_2d_norm = feature_2d_norm/feature_2d_norm.max() #[H, W], data is converted 0~1 value.
    feature_2d_norm = feature_2d_norm*contrast + brightness

    ## Channel attention
    ch_weight = GAP(feature)  # GAP(General Avg Pooling), [512, H, W] ==> [512, 1, 1], channel attention mask
    softmax = torch.nn.Softmax(dim=0)
    ch_weight_norm = softmax(ch_weight.squeeze().squeeze())
    ch_weight_norm = ch_weight_norm-ch_weight_norm.min()
    ch_weight_norm = ch_weight_norm/ch_weight_norm.max()
    ch_weight_norm = ch_weight_norm.unsqueeze(-1).unsqueeze(-1)
    #feature_ch_att = feature * ch_weight  # self-attention in channel direction.  [512, H, W]
    feature_ch_att = feature * ch_weight_norm  # self-attention in channel direction.  [512, H, W]

    ## Spatial attention
    feature_2d_att = feature_ch_att.mean(dim=0) # [512, H, W] ==> [H, W]  # Average in channel direction for spatial attention.

    ## Spatial normalize
    if normEn:
        feature_2d_att = feature_2d_att - feature_2d_att.min()
        feature_2d_att = feature_2d_att/feature_2d_att.max()  # [H, W], data is converted 0~1 value.

    mask = torch.nn.functional.interpolate(feature_2d_att.unsqueeze(0).unsqueeze(0).float(), size=(480,640), mode="bicubic", align_corners=True) # ==> [1, 1, 480, 640]
    mask = mask.squeeze(0)  # ==> [1, 480, 640]
    mask = mask.expand(3,-1,-1).permute(1,2,0)  # [H, W, 3]
    mask = mask.detach().cpu().numpy()
    #mask = np.maximum(mask, 0)  # ReLU effect. Use positive value

    mask = np.abs(mask)  # ReLU effect. Use positive value
    denom = mask.max()
    if denom != 0:
        mask = mask / denom

    mask = mask*contrast + brightness
 
    img_jet = show_cam_on_image(imgname, mask, ofname)
    if dispEn:
        clf()
        imshow(img_jet, 131, 'img_att')
        imshow(feature_2d_ori, 132, 'Fea')
        imshow(feature_2d_att, 133, 'FeaAtt')

    return feature_2d_norm, feature_2d_att, mask, img_jet

def Normalize():
    '''
    Normalize using torch transform
    Usage:
        torch_img_norm = input_transform(torch_img)  # ori. to norm.
        torch_img = input_transform(torch_img_norm)  # norm. to ori.
    '''
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])  

class Denormalize(object):
    '''
    Usage :
    denorm = tim.Denormalize()
    newimg = denorm(img)
    '''
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)

from typing import Union, Optional, List, Tuple, Text, BinaryIO
import pathlib
import torch
import math
import warnings
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor

__all__ = ["make_grid", "save_image", "draw_bounding_boxes"]

@torch.no_grad()
def make_grid(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: int = 0,
    **kwargs
) -> torch.Tensor:
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        value_range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if "range" in kwargs.keys():
        warning = "range will be deprecated, please use value_range instead."
        warnings.warn(warning)
        value_range = kwargs["range"]

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if value_range is not None:
            assert isinstance(value_range, tuple), \
                "value_range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid


@torch.no_grad()
def save_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[Text, pathlib.Path, BinaryIO],
    format: Optional[str] = None,
    **kwargs
) -> None:
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)


@torch.no_grad()
def draw_bounding_boxes(
    image: torch.Tensor,
    boxes: torch.Tensor,
    labels: Optional[List[str]] = None,
    colors: Optional[List[Union[str, Tuple[int, int, int]]]] = None,
    fill: Optional[bool] = False,
    width: int = 1,
    font: Optional[str] = None,
    font_size: int = 10
) -> torch.Tensor:

    """
    Draws bounding boxes on given image.
    The values of the input image should be uint8 between 0 and 255.
    If filled, Resulting Tensor should be saved as PNG image.

    img = torch.rand((3,480,640)).to(dtype=torch.uint8)
    boxes = torch.Tensor([[10,10, 30, 40]])
    init()
    imshow(draw_bounding_boxes(img, boxes))

    Args:
        image (Tensor): Tensor of shape (C x H x W)
        boxes (Tensor): Tensor of size (N, 4) containing bounding boxes in (xmin, ymin, xmax, ymax) format. Note that
            the boxes are absolute coordinates with respect to the image. In other words: `0 <= xmin < xmax < W` and
            `0 <= ymin < ymax < H`.
        labels (List[str]): List containing the labels of bounding boxes.
        colors (List[Union[str, Tuple[int, int, int]]]): List containing the colors of bounding boxes. The colors can
            be represented as `str` or `Tuple[int, int, int]`.
        fill (bool): If `True` fills the bounding box with specified color.
        width (int): Width of bounding box.
        font (str): A filename containing a TrueType font. If the file is not found in this filename, the loader may
            also search in other directories, such as the `fonts/` directory on Windows or `/Library/Fonts/`,
            `/System/Library/Fonts/` and `~/Library/Fonts/` on macOS.
        font_size (int): The requested font size in points.
    """

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Tensor expected, got {type(image)}")
    elif image.dtype != torch.uint8:
        raise ValueError(f"Tensor uint8 expected, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")

    ndarr = image.permute(1, 2, 0).numpy()
    img_to_draw = Image.fromarray(ndarr)

    img_boxes = boxes.to(torch.int64).tolist()

    if fill:
        draw = ImageDraw.Draw(img_to_draw, "RGBA")

    else:
        draw = ImageDraw.Draw(img_to_draw)

    txt_font = ImageFont.load_default() if font is None else ImageFont.truetype(font=font, size=font_size)

    for i, bbox in enumerate(img_boxes):
        if colors is None:
            color = None
        else:
            color = colors[i]

        if fill:
            if color is None:
                fill_color = (255, 255, 255, 100)
            elif isinstance(color, str):
                # This will automatically raise Error if rgb cannot be parsed.
                fill_color = ImageColor.getrgb(color) + (100,)
            elif isinstance(color, tuple):
                fill_color = color + (100,)
            draw.rectangle(bbox, width=width, outline=color, fill=fill_color)
        else:
            draw.rectangle(bbox, width=width, outline=color)

        if labels is not None:
            draw.text((bbox[0], bbox[1]), labels[i], fill=color, font=txt_font)

    return torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1)
