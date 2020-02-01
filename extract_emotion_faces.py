import json
import numpy as np
import os
import sys
import shutil


def extract_emotion_pics():
    files_dir = ['result-2', 'result-2-2']
    face_info_path = 'face_info.txt'
    out_dir = 'labeled_emotion_face'
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    os.mkdir(os.path.join(out_dir, 'generate_code'))
    with open(os.path.join(out_dir, 'face_info.txt'), 'w') as fw:
        for f_dir in files_dir:
            with open(os.path.join(f_dir, face_info_path)) as fr:
                for line in fr:
                # {"iamge_path": "result-2/000009.png", "face_token": "ceb7c193c1a6365400256305e472df5f", "emotion": {"type": "neutral", "probability": 0.99}, "expression": {"type": "none", "probability": 1}}
                    info = json.loads(line.strip())
                    file_path = info['iamge_path']
                    file_path_out = file_path.replace('/', '-')
                    os.system('cp {} {}'.format(file_path, os.path.join(out_dir, file_path_out)))
                    os.system('cp {} {}'.format(os.path.join(file_path.split('/')[0], 'generate_code', file_path.split('/')[-1].replace('png', 'txt')), os.path.join(out_dir, 'generate_code', file_path_out.replace('png', 'txt'))))
                    info['iamge_path'] = file_path_out
                    fw.write(json.dumps(info) + '\n')

if __name__ == '__main__':
    extract_emotion_pics()
