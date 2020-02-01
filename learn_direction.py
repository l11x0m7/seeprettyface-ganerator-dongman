import requests
import base64
import os
import sys
import pickle
import numpy as np
import PIL.Image
from PIL import Image
import dnnlib.tflib as tflib
import random
import time
import json
import cv2
import numpy as np
from resnet import resnet50
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
import dnnlib
import dnnlib.tflib as tflib
from matplotlib import pyplot as plt

np.random.seed(1314)
random.seed(1314)

def load_txt(filepath):
    with open(filepath) as fr:
        arr = []
        for line in fr:
            arr.append(float(line.strip()))
    return np.asarray(arr)


def generate_image(Gs, latent_vector, outpath):
    latent_vector = latent_vector.reshape((1, 512))
    # generator.set_dlatents(latent_vector)
    # img_array = generator.generate_images()[0]
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    img_array = Gs.run(latent_vector, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)[0]
    img = PIL.Image.fromarray(img_array, 'RGB')
    img.save(outpath)
    return img

def move_and_show(generator, latent_vector, direction, coeffs, outdir, fname):
    fig,ax = plt.subplots(1, len(coeffs), figsize=(15, 10), dpi=80)
    for i, coeff in enumerate(coeffs):
        new_latent_vector = latent_vector.copy()
        new_latent_vector[:8] = (latent_vector + coeff*direction)[:8]
        ax[i].imshow(generate_image(generator, new_latent_vector, os.path.join(outdir, '{}_'.format(coeff) + fname)))
        ax[i].set_title('Coeff: %0.1f' % coeff)
    [x.axis('off') for x in ax]
    plt.savefig(os.path.join(outdir, 'comp_' + fname))
    # plt.show()

def extract_latent_features(dirpath):
    with open(os.path.join(dirpath, 'face_info.txt')) as fr:
        model = resnet50(512)
        model.load_weights('model/encoder_model_1000000.h5')
        X_data = []
        y_data = []
        img_names = []
        counter = 0
        for line in fr:
            # {"iamge_path": "result-2-000009.png", "face_token": "ceb7c193c1a6365400256305e472df5f", "emotion": {"type": "neutral", "probability": 0.99}, "expression": {"type": "none", "probability": 1}}
            info = json.loads(line.strip())
            if info['emotion']['type'] == '':
                continue
            img_path = info['iamge_path']
            img_name = img_path.split('/')[-1]
            ori_lv = load_txt(os.path.join(dirpath, 'generate_code', img_path.replace('png', 'txt')))
            img = np.asarray(Image.open(os.path.join(dirpath, img_path)))
            img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC) / 128.
            # img = Image.fromarray(img.astype('uint8')).convert('RGB')
            latent_vector = model.predict_on_batch(np.asarray([img]))
            # X_data.append(latent_vector[0].flatten())
            X_data.append(ori_lv)
            # print(ori_lv - latent_vector)
            y_data.append(info['emotion']['type'] == 'happy')
            img_names.append(img_name)
            counter += 1
            if counter == 100:
                break
        return X_data, y_data, img_names

def train_latent_direction(dirpath):
    X_data, y_data, img_names = extract_latent_features(dirpath)
    X_data = np.asarray(X_data)
    y_data = np.asarray(y_data)
    clf = LogisticRegression(class_weight='balanced').fit(X_data, y_data)
    emotion_direction = clf.coef_.reshape((512, ))
    np.save(os.path.join('emotion_direction', 'happy.npy'), emotion_direction)
    emotion_direction = np.load(os.path.join('emotion_direction', 'happy.npy'))

    model_path = 'model/generator_dongman.pkl'
    tflib.init_tf()
    with open(model_path, "rb") as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)

    # generator = Generator(Gs_network, batch_size=1, randomize_noise=False)
    for i in range(10):
        move_and_show(Gs_network, X_data.reshape((-1, 512))[i], emotion_direction, [-40, -20, -10, -1.5, 0, 1.5, 10, 20, 40], os.path.join(dirpath, 'generated_emotion_face'), img_names[i])


if __name__ == '__main__':
    dirpath = 'labeled_emotion_face' 
    train_latent_direction(dirpath)

