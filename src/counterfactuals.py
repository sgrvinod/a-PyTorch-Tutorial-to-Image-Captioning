import sys
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import tensorflow as tf
from tqdm import tqdm
import os
import cv2
from zipfile import ZipFile
import time
import neuralgym as ng
sys.path.insert(0, 'src/')
from download import *
from inpaint_model import InpaintCAModel
print(os.getcwd())
sys.path.insert(0, 'src/data/cocoapi/PythonAPI/')
from pycocotools.coco import COCO

def dir_creator(dir_name):
    try:
        os.chdir(dir_name)
        for _ in dir_name.split('/'):
            os.chdir('..')
    except FileNotFoundError:
        if len(dir_name.split('/')) > 1:
            tot_dir = ''
            for dir in dir_name.split('/'):
                tot_dir += dir
                try:os.mkdir(tot_dir)
                except FileExistsError:pass
                tot_dir += '/'
        else:
            os.mkdir(dir_name)

def download_gen_inpaint_model(file_id, model_dir):
    # file_id = '11fD6YYG4kL1WT_a27LSOl5tOZOsg7TOM'

    # dir_creator(model_dir.split('/')[0])
    destination = f'{model_dir}.zip'
    print(f'Downloading file at {destination}...')
    print(file_id)
    download_file_from_google_drive(file_id, destination)
    with ZipFile(f'{model_dir}.zip', 'r') as z:
        os.chdir(model_dir.split('/')[0])
        z.extractall()
        print(f'extracted to {model_dir}.zip')
        # time.sleep(100)
        os.chdir('..')


def create_mask_input(data_dir, temp_dir, out_dir, annotation_fp, input=False):
    # dir_creator(temp_dir)
    print(os.getcwd())
    coco = COCO(annotation_fp)
    for i in tqdm(range(len(os.listdir(data_dir)))):
        img_id = os.listdir(data_dir)[i]
        img_id = int(img_id.strip('.jpg').split('_')[-1])
    #     print(img_id)
        fname = coco.loadImgs(img_id)[0]['file_name']
    #     print(fname)
    #     time.sleep(100)
        ann_ids = coco.getAnnIds(img_id)
        print(fname)
        print(ann_ids)
    #     ann_id = ann_ids[get_random_idx(ann_ids)]
        for ann_id in ann_ids:
            ann = coco.loadAnns(ann_id)[0]
            mask = coco.annToMask(ann) * 255

            img_dir = str(img_id)
            #try:
            #    print('current directory:',os.getcwd())

            #    os.chdir(f'{temp_dir}/{img_dir}')
            #    os.chdir('../../../../')
            #except FileNotFoundError:
            #    print('current directory:',os.getcwd())
            #    #if os.path.exists(f'{temp_dir}')==False: #or os.path.exists(f'{out_dir}')==False:
            #    #    dir_tot = ''
            #    #    for dir in f'{temp_dir}'.split('/'):
            #    #        dir_tot += dir
            #    #        os.mkdir(dir_tot)
            #    #        dir_tot += '/'
            #    #    #for dir in f'{out_dir}'.split('/'):
            #    #    #    os.mkdir(dir
            dir_creator(f'{temp_dir}/{img_dir}')
            # os.mkdir(f'{out_dir}/{img_dir}')
            print(f'{temp_dir}/{img_dir} directory created')

            print('current directory:',os.getcwd())

            mask = Image.fromarray(mask)
            mask_fname = f'{temp_dir}/{img_id}/mask_{ann_id}.png'
            print(os.getcwd())
            mask.save(mask_fname)
            print(f'{mask_fname} saved')


            img = Image.open(f'{data_dir}/{fname}')
            raw_fname = f'{temp_dir}/{img_id}/raw_{img_id}.png'
            img.save(raw_fname)
            print(f'{raw_fname} saved')
            if input == True:
                raw_cv = cv2.imread(f'{temp_dir}/{img_id}/raw_{img_id}.png')
                raw_cv = cv2.cvtColor(raw_cv, cv2.COLOR_BGR2RGB)
                mask_cv = cv2.imread(mask_fname)

                input_cv = cv2.add(raw_cv, mask_cv)
                input_cv = Image.fromarray(input_cv)
                input_fname = f'{temp_dir}/{img_id}/input_{ann_id}.png'
                input_cv.save(input_fname)
                print(f'{input_fname} saved\n')
        #     os.chdir('..')


def generate_counterfactual(image_fp, mask_fp, output_fp, checkpoint_dir, model_id=None):
    try:FLAGS = ng.Config('config/inpaint.yml')
    except AssertionError:
        print(os.getcwd())
        raise ValueError('check directory above')
    # ng.get_gpus(1)
    # args, unknown = parser.parse_known_args()

    model = InpaintCAModel()
    image = cv2.imread(image_fp)
    mask = cv2.imread(mask_fp)
    # mask = cv2.resize(mask, (0,0), fx=0.5, fy=0.5)

    assert image.shape == mask.shape

    h, w, _ = image.shape
    grid = 8
    image = image[:h//grid*grid, :w//grid*grid, :]
    mask = mask[:h//grid*grid, :w//grid*grid, :]
    print('Shape of image: {}'.format(image.shape))

    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        input_image = tf.constant(input_image, dtype=tf.float32)
        output = model.build_server_graph(FLAGS, input_image)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        print('Model loaded.')
        result = sess.run(output)
        cv2.imwrite(output_fp, result[0][:, :, ::-1])
        print(f'IMAGE WROTE TO {output_fp}\n\n\n')
    tf.reset_default_graph()





def generate_counterfactuals(data_dir, checkpoint_dir, model_id=None):
    dir_creator(checkpoint_dir.split('/')[0]) # create models directory if it doesn't already exist
    try:
        os.chdir(checkpoint_dir)
        print(f'Model exists at {checkpoint_dir}')
        os.chdir('../..')
    except FileNotFoundError:
        download_gen_inpaint_model(model_id, checkpoint_dir)
    for img_id in os.listdir(data_dir):
        img_dir = f'{data_dir}/{img_id}'
        for file in os.listdir(img_dir):
            if 'raw' in file:
                input_fp = f'{img_dir}/{file}'
                break
        for file in os.listdir(img_dir):
            if 'mask' in file:
                mask_fp = f'{img_dir}/{file}'
                ann_id = file.strip('.png').split('_')[-1]
                dir_creator(f'{img_dir}/counterfactuals')
                while os.getcwd().endswith('180b_capstone_xai') == False:
                    os.chdir('..')
                # os.chdir('../../../../')
                output_fp = f'{img_dir}/counterfactuals/cf_{ann_id}.png'
                generate_counterfactual(input_fp, mask_fp, output_fp, checkpoint_dir)

# def generate_counterfactual(image_fp, mask_fp, output_fp, checkpoint_dir):


# if __name__ == "__main__":
#     generate_counterfactuals(data_dir='data/raw_images', temp_dir='../data/temp/imgs', out_dir='../data/out', val_fp='../data/raw/annotations/instances_val2014.json')
