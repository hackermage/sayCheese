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

def crop_align_decode(json_data):
    imageData = base64.b64decode(json_data['image_key'].encode())
    npimg = np.fromstring(imageData, dtype=np.uint8)
    img_raw = cv2.imdecode(npimg, 1) 
    img_raw = cv2.resize(img_raw,(0,0), fx=1, fy=1)
    img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    real_face, face_origin_pos = face.face_crop_and_align(img)
    return real_face, face_origin_pos

def realign_encode(img_raw, processed_face, face_origin_pos):
    new_img = face.face_place_back(img_raw, processed_face, face_origin_pos)
    img_str = cv2.imencode('.jpg', new_img)[1].tostring()
    base64_bytes = base64.b64encode(img_str)
    base64_string = base64_bytes.decode('utf-8')
    return base64_string

#configure flask
app.config['TESTING'] = True
model_path = 'checkpoints/model_align/'
load_epoch = 40

# load pretrained GANimation model and run
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

        real_face, face_origin_pos = crop_align_decode(json_data):

        expressions = np.ndarray((5,17), dtype = np.float)

        # define the target AU, for test use only
        target_AU = np.array([0.25, 0.11, 0.2 , 0.16, 1.92, 1.03, 0.3 , 2.15, 2.88, 1.61, 0.03, 0.09, 0.16, 0.11, 2.25, 0.37, 0.05], dtype = np.float)

        # find original AU of input image using discriminator of GANimation, for test use only
        try:
            out_real, out_aux = convertor.FindAU(real_face) # out_aux is the AU value from D 
        except:
            return {'status_code': 100 }

        original_AU = out_aux.data.numpy()
        c = (original_AU - target_AU)/4
        expressions = c * 4 + target_AU
        processed_face, maskA = convertor.Foward(real_face, expressions)

        base64_img_str = realign_encode(img_raw, processed_face, face_origin_pos)
        
        return {'image_key': base64_img_str, 'status_code': 0}

api.add_resource(make_cheese, '/api')

if __name__ == '__main__':
    app.debug = True
    app.run()
