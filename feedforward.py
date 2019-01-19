import argparse
import os
import glob
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch
import cv2
import numpy as np
from networks import cruzhack_forward
import utils.util as util
import utils.cruzhack_utils as face
import pickle
import calendar
import time
import random

class feedFoward:
    def __init__(self, pathG, pathD):
        # load pre-trained generator here
        self._modelG = cruzhack_forward.generatorFoward(conv_dim=64, c_dim=17, repeat_num=6)
        self._modelG.load_state_dict(torch.load(pathG, map_location='cpu'))
        self._modelG.eval()
        self._transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                   std=[0.5, 0.5, 0.5])
                                              ])
        #load pre-trained discriminator here
        self._modelD = cruzhack_forward.Discriminator(image_size=128, conv_dim=64, c_dim=17, repeat_num=6)
        self._modelD.load_state_dict(torch.load(pathD, map_location='cpu'))
        self._modelD.eval()

    def Foward(self, face, desired_expression):
        # transform face by normalizing 
        face = torch.unsqueeze(self._transform(face), 0).float()
        desired_expression = torch.unsqueeze(torch.from_numpy(desired_expression/5.0), 0).float() # set up desired expression

        color, mask = self._modelG.forward(face, desired_expression)
        # calculate result
        masked_result = mask * face + (1.0 - mask) * color

        img = self.convertToimg(masked_result)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)# convert to OpenCV color BGR

        maskA = mask.detach().numpy()[0]
        maskA = np.transpose(maskA, (1,2,0))

        return img, maskA

    def FindAU(self, face):
        # transform face by normalizing 
        face = torch.unsqueeze(self._transform(face), 0).float()
        out_real, out_aux = self._modelD.forward(face)
        return out_real, out_aux
    
    def convertToimg(self, tensor):
        tensor = tensor.cpu().float()
        img = tensor.detach().numpy()[0]
        img = img*0.5+0.5 # reverse transform
        img = np.transpose(img, (1, 2, 0))
        return img*254.0

def find_epoch(model_path, load_epoch_num):
    if os.path.exists(model_path):
        if load_epoch_num == -1:
            epoch_num = 0
            for file in os.listdir(model_path):
                if file.startswith("net_epoch_"):
                    epoch_num = max(epoch_num, int(file.split('_')[2]))
        else:
            found = False
            for file in os.listdir(model_path):
                if file.startswith("net_epoch_"):
                    found = int(file.split('_')[2]) == load_epoch_num
                    if found: break
            assert found, 'Model for epoch %i not found' % load_epoch_num
            epoch_num = load_epoch_num
    else:
        assert load_epoch_num < 1, 'Model for epoch %i not found' % load_epoch_num
        epoch_num = 0

    return epoch_num

# =========================================================== #
# below, the main(), is a demo for how to use the feedforward #
# =========================================================== #
def main():
    parser = argparse.ArgumentParser(description='easy for input parameters')
    parser.add_argument('--img_path', type=str, 
                        default='/Users/xyli1905/Projects/Datasets/imgs_178/000009.png', 
                        help='path to the test image')
    parser.add_argument('--model_path', type=str, default='./checkpoints/model_align/', 
                        help='path to the pretrained model')
    parser.add_argument('--load_epoch', type=int, default=-1, help='specify the model to be loaded')

    parser.add_argument('--AU', type=str, default = './dataset/aus_openface.pkl', 
                        help = 'loading pre-processing AU')

    arg = parser.parse_args()

    AU_file = open(arg.AU, 'rb')
    conds = pickle.load(AU_file)

    image_name = arg.img_path.split('.')[-2]
    print(image_name)
    image_name = image_name.split('/')[-1]

    # use any original image as you want and clip it
    img_raw = cv2.imread(arg.img_path)
    img_raw = cv2.resize(img_raw,(0,0), fx=1, fy=1)
    img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

    real_face, face_origin_pos = face.face_crop_and_align(img)

    # load pretrained GANimation model and run
    epoch_num = find_epoch(arg.model_path, arg.load_epoch)
    load_filename_generator = 'net_epoch_%s_id_G.pth' % (epoch_num)
    load_filename_discriminator = 'net_epoch_%s_id_D.pth' % (epoch_num)
    pathG = os.path.join(arg.model_path, load_filename_generator)
    pathD = os.path.join(arg.model_path, load_filename_discriminator)

    convertor = feedFoward(pathG, pathD)

    # define the target AU, for test use only
    # target_AU = np.array(
    # [[0.2 , 0.12, 0.1 , 0.24, 0.05, 0.12, 0.08, 0.11, 0.19, 0.22, 0.1 , 0.09, 0.15, 0.07, 0.11, 0.16, 0.03],
    #  [0.4 , 0.1 , 0.28, 0.45, 0.46, 0.32, 0.36, 1.61, 0.72, 0.45, 0.34, 0.28, 0.29, 0.09, 0.34, 0.3 , 0.07],
    #  [0.51, 0.37, 0.17, 1.53, 0.09, 0.16, 0.1 , 0.21, 0.21, 0.21, 0.19, 0.15, 0.2 , 0.09, 0.23, 0.29, 0.04],
    #  [1.02, 0.34, 2.  , 0.54, 0.32, 0.66, 0.25, 0.6 , 0.38, 0.56, 0.33, 0.33, 0.3 , 0.13, 0.28, 0.36, 0.09],
    #  [0.27, 0.17, 0.08, 0.26, 0.51, 0.38, 0.12, 0.57, 1.46, 0.56, 0.04, 0.04, 0.16, 0.06, 1.22, 0.44, 0.03],
    #  [0.27, 0.2 , 0.14, 0.26, 0.07, 0.19, 0.07, 0.23, 0.2 , 0.23, 0.16, 0.1 , 0.2 , 0.06, 0.43, 1.19, 0.05],
    #  [1.46, 0.76, 0.5 , 0.79, 1.15, 0.95, 0.2 , 1.48, 1.94, 1.04, 0.11, 0.13, 0.3 , 0.09, 1.79, 0.53, 0.06],
    #  [0.67, 0.35, 0.34, 0.51, 0.47, 0.7 , 0.26, 0.84, 0.54, 0.5 , 0.4 , 0.26, 0.33, 0.14, 1.14, 2.12, 0.09],
    #  [1.52, 0.8 , 0.24, 0.41, 0.15, 0.21, 0.07, 0.23, 0.3 , 0.32, 0.23, 0.19, 0.32, 0.09, 0.25, 0.29, 0.05],
    #  [2.07, 1.32, 0.49, 1.72, 0.4 , 0.38, 0.11, 0.56, 0.56, 0.65, 0.33, 0.54, 0.42, 0.2 , 0.39, 0.39, 0.06],
    #  [0.38, 0.13, 0.27, 0.32, 0.4 , 1.5 , 0.27, 0.25, 0.45, 0.31, 0.23, 0.15, 0.37, 0.12, 0.28, 0.37, 0.04],
    #  [0.92, 0.34, 0.75, 0.66, 1.47, 2.64, 0.69, 0.98, 0.84, 0.61, 0.56, 0.72, 0.91, 0.28, 0.47, 0.34, 0.05],
    #  [0.23, 0.07, 0.18, 0.16, 1.04, 1.66, 0.28, 0.68, 1.64, 0.96, 0.07, 0.13, 0.31, 0.16, 0.82, 0.34, 0.04],
    #  [0.17, 0.1 , 0.1 , 0.17, 1.11, 0.48, 0.15, 1.35, 2.3 , 1.19, 0.02, 0.05, 0.13, 0.08, 1.7 , 0.3 , 0.03],
    #  [0.26, 0.13, 0.13, 0.22, 0.3 , 0.26, 0.12, 0.4 , 0.96, 1.37, 0.07, 0.27, 0.24, 0.21, 0.21, 0.25, 0.04],
    #  [0.65, 0.29, 0.33, 0.49, 0.67, 0.44, 0.27, 0.8 , 0.56, 0.74, 0.62, 1.73, 0.6 , 0.44, 0.11, 0.38, 0.08],
    #  [0.25, 0.11, 0.2 , 0.16, 1.92, 1.03, 0.3 , 2.15, 2.88, 1.61, 0.03, 0.09, 0.16, 0.11, 2.25, 0.37, 0.05],
    #  [0.34, 0.1 , 0.48, 0.23, 1.44, 0.89, 0.68, 1.84, 1.63, 1.7 , 0.22, 0.63, 0.34, 0.32, 0.53, 0.32, 0.07],
    #  [0.19, 0.06, 0.16, 0.14, 1.59, 1.9 , 0.27, 1.32, 2.38, 0.92, 0.04, 0.03, 0.2 , 0.08, 1.98, 0.45, 0.04],
    #  [0.32, 0.06, 0.49, 0.17, 2.37, 2.65, 0.71, 2.16, 2.65, 1.4 , 0.08, 0.07, 0.32, 0.13, 2.29, 0.55, 0.05]], dtype = np.float)

    target_AU = np.array(
    [[0.3 , 0.18, 0.1 , 0.25, 0.8 , 0.46, 0.13, 0.94, 1.89, 0.87, 0.03, 0.05, 0.15, 0.07, 1.49, 0.39, 0.03],
     [0.22, 0.09, 0.1 , 0.17, 1.46, 1.83, 0.27, 1.16, 2.23, 0.9 , 0.05, 0.04, 0.22, 0.08, 1.78, 0.44, 0.04],
     [2.08, 1.28, 0.48, 1.6 , 0.51, 0.43, 0.12, 0.7 , 0.75, 0.77, 0.31, 0.53, 0.43, 0.2 , 0.55, 0.44, 0.07],
     [0.24, 0.06, 0.25, 0.14, 2.31, 2.31, 0.57, 2.15, 2.8 , 1.43, 0.05, 0.07, 0.26, 0.12, 2.35, 0.51, 0.05],
     [0.46, 0.29, 0.21, 0.38, 0.21, 0.39, 0.14, 0.47, 0.35, 0.33, 0.27, 0.15, 0.26, 0.09, 0.81, 1.71, 0.07],
     [0.19, 0.12, 0.1 , 0.23, 0.05, 0.15, 0.08, 0.11, 0.19, 0.21, 0.1 , 0.09, 0.16, 0.06, 0.16, 0.28, 0.04],
     [0.86, 0.24, 1.63, 0.36, 1.64, 2.21, 0.59, 1.66, 1.88, 1.1 , 0.14, 0.12, 0.32, 0.13, 1.71, 0.54, 0.07],
     [0.31, 0.15, 0.16, 0.21, 1.6 , 0.71, 0.22, 1.92, 2.69, 1.47, 0.03, 0.08, 0.16, 0.1 , 2.09, 0.34, 0.04],
     [0.59, 0.27, 0.33, 0.47, 0.6 , 0.39, 0.27, 0.78, 0.52, 0.75, 0.61, 1.68, 0.54, 0.43, 0.1 , 0.39, 0.08],
     [0.32, 0.11, 0.22, 0.25, 0.58, 1.56, 0.27, 0.34, 0.83, 0.5 , 0.15, 0.14, 0.35, 0.14, 0.44, 0.38, 0.04],
     [0.41, 0.1 , 0.28, 0.44, 0.45, 0.32, 0.35, 1.57, 0.73, 0.42, 0.32, 0.25, 0.28, 0.08, 0.41, 0.33, 0.07],
     [0.24, 0.12, 0.12, 0.22, 0.29, 0.26, 0.11, 0.36, 1.  , 1.27, 0.06, 0.22, 0.23, 0.19, 0.26, 0.26, 0.03],
     [1.  , 0.33, 1.94, 0.54, 0.3 , 0.66, 0.25, 0.57, 0.34, 0.52, 0.33, 0.32, 0.3 , 0.13, 0.26, 0.37, 0.09],
     [0.35, 0.11, 0.38, 0.23, 1.36, 0.91, 0.6 , 1.67, 1.63, 1.66, 0.2 , 0.56, 0.34, 0.3 , 0.51, 0.31, 0.07],
     [1.49, 0.81, 0.23, 0.45, 0.14, 0.22, 0.07, 0.21, 0.28, 0.29, 0.23, 0.18, 0.31, 0.08, 0.25, 0.29, 0.05],
     [0.5 , 0.37, 0.17, 1.52, 0.09, 0.18, 0.1 , 0.21, 0.22, 0.22, 0.19, 0.15, 0.21, 0.09, 0.23, 0.3 , 0.04],
     [0.62, 0.2 , 0.64, 0.43, 1.08, 1.89, 0.53, 0.91, 0.88, 0.7 , 0.4 , 0.56, 0.6 , 0.25, 0.48, 0.44, 0.06],
     [0.27, 0.15, 0.17, 0.36, 0.13, 0.25, 0.12, 0.27, 0.34, 0.39, 0.15, 0.18, 0.2 , 0.1 , 0.24, 0.42, 0.04],
     [0.31, 0.15, 0.17, 0.24, 0.9 , 0.57, 0.21, 1.12, 1.8 , 1.1 , 0.07, 0.17, 0.19, 0.13, 1.19, 0.39, 0.04],
     [1.52, 0.84, 0.55, 1.08, 0.28, 0.31, 0.11, 0.5 , 0.43, 0.51, 0.32, 0.43, 0.37, 0.15, 0.35, 0.44, 0.06],
     [0.29, 0.1 , 0.28, 0.18, 1.89, 1.73, 0.39, 1.84, 2.61, 1.3 , 0.05, 0.08, 0.22, 0.11, 2.12, 0.44, 0.04],
     [0.92, 0.35, 0.58, 0.68, 1.43, 2.54, 0.64, 0.91, 0.8 , 0.58, 0.6 , 0.8 , 0.98, 0.3 , 0.39, 0.35, 0.05]], dtype = np.float)
    # target_AU = np.array([0.25, 0.11, 0.2 , 0.16, 1.92, 1.03, 0.3 , 2.15, 2.88, 1.61, 0.03, 0.09, 0.16, 0.11, 2.25, 0.37, 0.05], dtype = np.float)

    # # find original AU of input image using discrinator of GANimation, for test use only
    out_real, out_aux = convertor.FindAU(real_face) # out_aux is the AU value from D 
    original_AU = out_aux.data.numpy()

    # set expression
    row_num = np.shape(target_AU)[0]

    expression_num = 5
    expressions = np.ndarray((expression_num, row_num, 17), dtype = np.float)

    # # interpolate between original_AU and the target_AU, for test only
    o_AU = np.repeat([original_AU], row_num, axis = 0)
    assert np.shape(o_AU) == np.shape(target_AU), "shape not match"

    c = (o_AU - target_AU)/(expression_num - 1)
    for i in range(expression_num):
        expressions[i] = c * i + target_AU

    # run model for above expressions
    result = np.zeros((1, img_raw.shape[1] * (expression_num + 1), 3))
    for j in range(row_num):
        current_result = img_raw
        for i in range(expression_num):
            processed_face, maskA = convertor.Foward(real_face, expressions[i, j])
            new_img, mask, rotate, new_maskA = face.face_place_back(img_raw, processed_face, face_origin_pos, 
                                                                    maskA_test=True, maskA = maskA)
            current_result = np.hstack((current_result, new_img))
        result = np.vstack((result, current_result))

    #     processed_face, maskA = convertor.Foward(real_face, expressions[i])
    #     new_img, mask, rotate, new_maskA = face.face_place_back(img_raw, processed_face, face_origin_pos, 
    #                                                             maskA_test=True, maskA = maskA)

    #     # handle maskA to image for dispalying
    #     maskA_t = np.expand_dims(new_maskA, axis=2)
    #     maskA_t = (maskA_t*254.0).astype(np.uint8)
    #     maskA_t = np.repeat(maskA_t, 3, axis=-1)

    #     # wrap up results
    #     current_result = np.hstack((img_raw, rotate, maskA_t, new_img))
    #     result = np.vstack((result, current_result))
    #     #print(np.shape(processed_face), np.shape(maskA), maskA.dtype)

    #result  = cv2.resize(result,(0,0), fx=0.5, fy=0.5)

    timestamp = calendar.timegm(time.gmtime())
    image_out_name = "./results/processedface_"+image_name+"-"+str(timestamp)+".jpg"
    cv2.imwrite(image_out_name, result)
    print("Processed image saved as %s" % image_out_name)
    #cv2.imshow('result', result/254.0)
    #cv2.waitKey()
'''
    while True:
        key = cv2.waitKey()
        if key == 27:
            break;
'''

if __name__ == '__main__':
    main()
#