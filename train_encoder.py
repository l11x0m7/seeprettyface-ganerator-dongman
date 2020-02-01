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
import cv2
import numpy as np
from resnet import resnet50

np.random.seed(1314)
random.seed(1314)

synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

def main():
    # Initialize TensorFlow.
    tflib.init_tf()

    # Load pre-trained network.
    model_path = 'model/generator_dongman.pkl'

    # Prepare result folder
    result_dir = 'encoder/'
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(result_dir + '/generate_code', exist_ok=True)
    with open(model_path, "rb") as f:
        _G, _D, Gs = pickle.load(f, encoding='latin1')

    # Print network details.
    Gs.print_layers()

    # Generate pictures
    batch_size = 32
    num_steps = 1000000
    model = resnet50(Gs.input_shape[1])
    if os.path.exists('model/encoder_model_400000.h5'):
        model.load_weights('model/encoder_model_400000.h5')
    # model.load_weights('model/home/hongkuny/hongkuny_keras_resnet50_gpu_8_fp32_eager_graph_cfit/checkpoints/model.ckpt-0090')
    model.compile(
            loss='mean_squared_error',
            optimizer='sgd',)
    print(model.summary())
    for _ in range(num_steps):
        latents = np.random.randn(batch_size, Gs.input_shape[1])

        # Generate image.
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
        images = np.stack([cv2.resize(__, dsize=(224, 224), interpolation=cv2.INTER_CUBIC) for __ in images]) / 128.
        # print(images.shape)
        # _, loss = sess.run([model.loss_op, model.loss], feed_dict={'image':images})
        loss = model.train_on_batch(images, latents)
        _ += 1
        if _ % 1000 == 0:
            print('step:{}, loss:{}'.format(_, loss))
        if _ % 100000 == 0:
            model.save('model/encoder_model_{}.h5'.format(_))

if __name__ == "__main__":
    main()

