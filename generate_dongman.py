# Thanks to StyleGAN provider —— Copyright (c) 2019, NVIDIA CORPORATION.
# Thanks to original dataset provider: https://www.gwern.net/Danbooru2018
# This work is trained by Copyright(c) 2018, seeprettyface.com, BUPT_GWY.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import requests
import base64
import os
import sys
import pickle
import numpy as np
import PIL.Image
import dnnlib.tflib as tflib
import random
import time
import json
np.random.seed(43210)
random.seed(43210)

synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

def text_save(file, data):  # save generate code, which can be modified to generate customized style
    for i in range(len(data[0])):
        s = str(data[0][i])+'\n'
        file.write(s)


def image_to_base64(iamge_path):
    with open(iamge_path, 'rb') as f:
        base64_data = base64.b64encode(f.read())
        s = base64_data.decode()
        return s

def get_access_token():
    # client_id 为官网获取的AK， client_secret 为官网获取的SK
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=usIUfrMlrYtO8YjqgGKxG0Wo&client_secret=20ynYqECVc8oI1S4LHZvARrV95qXmMZI'
    response = requests.get(host)
    # if response:
    #     print(response.json())
    access_token = response.json()['access_token']
    return access_token

def get_face_info(image_name, access_token):
    '''
    人脸检测与属性分析
    '''
    request_url = "https://aip.baidubce.com/rest/2.0/face/v3/detect"

    image_base64 = image_to_base64(image_name)

    params = "{\"image\":\"%s\",\"image_type\":\"BASE64\",\"face_field\":\"expression,emotion\"}" % image_base64
    request_url = request_url + "?access_token=" + access_token
    headers = {'content-type': 'application/json'}
    response = requests.post(request_url, data=params, headers=headers)
    # if response:
    #     print (response.json())
    return response.json()


def main():
    # Initialize TensorFlow.
    tflib.init_tf()

    # Load pre-trained network.
    model_path = 'model/generator_dongman.pkl'

    # Prepare result folder
    function_type = 2
    result_dir = 'result-{}-3'.format(function_type)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(result_dir + '/generate_code', exist_ok=True)

    with open(model_path, "rb") as f:
        _G, _D, Gs = pickle.load(f, encoding='latin1')

    # Print network details.
    Gs.print_layers()

    # Generate pictures
    
    if function_type == 1:
        generate_num = 20
        for k in range(Gs.input_shape[1]):
            latents = np.random.randn(1, Gs.input_shape[1])
            for i in range(generate_num):
        
                # Generate latent.
                latents[0, k] += 0.01 * (i - generate_num / 2)
        
                # Save latent.
                txt_filename = os.path.join(result_dir, 'generate_code/' + str(k).zfill(4) + '_' + str(i).zfill(4) + '.txt')
                file = open(txt_filename, 'w')
                text_save(file, latents)
        
                # Generate image.
                fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
                images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
        
                # Save image.
                png_filename = os.path.join(result_dir, str(k).zfill(4) + '_' + str(i).zfill(4) + '.png')
                PIL.Image.fromarray(images[0], 'RGB').save(png_filename)
        
                # Close file.
                file.close()

    elif function_type == 2:
        generate_num = 30000
        face_info_file = open(os.path.join(result_dir, 'face_info.txt'), 'a')
        access_token = get_access_token()
        for i in range(generate_num):
            latents = np.random.randn(1, Gs.input_shape[1])
            # Generate latent.
    
            # Save latent.
            txt_filename = os.path.join(result_dir, 'generate_code/' + str(i).zfill(6) + '.txt')
            file = open(txt_filename, 'w')
            text_save(file, latents)
    
            # Generate image.
            fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
            images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
    
            # Save image.
            png_filename = os.path.join(result_dir, str(i).zfill(6) + '.png')
            if os.path.exists(png_filename):
                continue

            PIL.Image.fromarray(images[0], 'RGB').save(png_filename)

            out = get_face_info(png_filename, access_token)
            print(out)
            if out['error_code'] == 0:
                out = out['result']['face_list'][0]
                if out['face_probability'] >= 0.2:
                    face_info = {'iamge_path': png_filename, 'face_token': out['face_token'], 'emotion':out['emotion'], 'expression':out['expression']}
                    print(face_info)
                    face_info_file.write(json.dumps(face_info) + '\n')
                    face_info_file.flush()
            sys.stdout.flush()
            # time.sleep(0.1)
        face_info_file.close()

if __name__ == "__main__":
    main()

