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
from flask import Flask, request
from flask_restful import Resource, Api
import feedforward
import base64

app = Flask(__name__)
api = Api(app)

def decode_img(json_data):
    imageData = base64.b64decode(json_data['image_key'].encode())
    npimg = np.fromstring(imageData, dtype=np.uint8)
    img_raw = cv2.imdecode(npimg, 1) 
    return img_raw

def encode_img(new_img):
    img_str = cv2.imencode('.jpg', new_img)[1].tostring()
    base64_bytes = base64.b64encode(img_str)
    base64_string = base64_bytes.decode('utf-8')
    return base64_string

#configure flask
app.config['TESTING'] = True
model_path = 'checkpoints/model_align/'
load_epoch = 40
list_len = 3

load pretrained GANimation model and run
epoch_num = feedforward.find_epoch(model_path, load_epoch)
load_filename_generator = 'net_epoch_%s_id_G.pth' % (epoch_num)
load_filename_discriminator = 'net_epoch_%s_id_D.pth' % (epoch_num)
pathG = os.path.join(model_path, load_filename_generator)
pathD = os.path.join(model_path, load_filename_discriminator)
convertor = feedforward.feedForward(pathG, pathD)

class make_cheese(Resource):
    def post(self):

        json_data = request.get_json(force=True)

        if not 'image_key' in json_data:
            return (json_data);

        img_raw = decode_img(json_data):

        # find original AU of input image using discriminator of GANimation, for test use only
        try:
            processed_img_dict = feedforward.img_processing(img_raw, convertor, list_len, test=False)
        except:
            return {'status_code': 100 }

        base64_img_str = encode_img(img_raw)
        
        for key in dict:
            processed_img_dict[key] = [encode_img(img) for img in processed_img_dict[key]]

        processed_img_dict['status_code'] = 0

        return processed_img_dict

        # return {'big_smile': [json['image_key'], json['image_key'], json['image_key']],
        #         'small_smile': [json['image_key'], json['image_key'], json['image_key']],
        #         'status_code': 0}

api.add_resource(make_cheese, '/api')

if __name__ == '__main__':
    app.debug = True
    app.run()
