import argparse
import os
import json
from math import ceil
from random import seed

import preprocessing.paths as P
import preprocessing.vizwiz as V
import preprocessing.utils as iu

from augmentation.image_main import image_transform


def create_input_files(dataset, image_folder, captions_per_image, min_word_freq, output_folder,
                       wordmap=False, max_len=100, text_augment=True, img_augment=True, nr_augs=10):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    if dataset == 'vizwiz':
        v = V.VizWiz(main_path=P.base_vizwiz)
        train_image_paths = v.train_paths
        val_image_paths = v.val_paths
        test_image_paths = v.test_paths
        train_image_captions = v.train_captions
        val_image_captions = v.val_captions
        test_image_captions = v.test_captions
        train_imgids = v.train_ids
        # for text augmentation
        aug_file = P.vizwiz_augs
        augXimg_file = P.vizwiz_augsXimg
    else:
        assert dataset in {'coco', 'flickr8k', 'flickr30k'}

        # Read Karpathy JSON
        with open(P.karpathy_coco, 'r') as j:
            data = json.load(j)

        # Read image paths and captions for each image
        train_image_paths = []
        train_image_captions = []
        val_image_paths = []
        val_image_captions = []
        test_image_paths = []
        test_image_captions = []

        for img in data['images']:
            captions = []
            for c in img['sentences']:
                # Update word frequency
                # word_freq.update(c['tokens'])
                if len(c['tokens']) <= max_len:
                    captions.append(c['raw'])  # NOTE

            if len(captions) == 0:
                continue

            path = os.path.join(image_folder, img['filepath'], img['filename']) if dataset == 'coco' else os.path.join(
                image_folder, img['filename'])

            if img['split'] in {'train', 'restval'}:
                train_image_paths.append(path)
                train_image_captions.append(captions)        
            elif img['split'] in {'val'}:
                val_image_paths.append(path)
                val_image_captions.append(captions)
            elif img['split'] in {'test'}:
                test_image_paths.append(path)
                test_image_captions.append(captions)

        # Sanity check
        assert len(train_image_paths) == len(train_image_captions)  # 113287
        assert len(val_image_paths) == len(val_image_captions)
        assert len(test_image_paths) == len(test_image_captions)

    if not wordmap:
        wordmap = input('Enter path to wordmap: ')

    with open(wordmap) as jsonfile:
        word_map = json.load(jsonfile)

        # word_map = dict()
        # word_map['<unk>'] = len(word_map) + 1
        # word_map['<start>'] = len(word_map) + 1
        # word_map['<end>'] = len(word_map) + 1
        # word_map['<pad>'] = 0
    # Or update it! wordmap should be path

    # Save word map to a JSON -- NOTE not relevant for now
    # with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        # json.dump(word_map, j)
    
    # Info for the writing of files
    info = dict()
    # Create a base/root name for all output files
    info['output_folder'] = output_folder
    info['base_filename'] = dataset + '_' + str(captions_per_image) \
        + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'
    info['captions_per_image'] = captions_per_image
    info['max_len'] = max_len
    
    # Dictionary for augmentation information storage
    extras = dict()
    # extras['train_image_paths_new'] = list()
    extras['train_imgids'] = train_imgids

    # text augmentations
    if text_augment:

        # Read augmentation files
        with open(aug_file) as j:
            augmentations = json.load(j)
        with open(augXimg_file) as js:
            augs_x_image = json.load(js)
        
        # Not all images are good enough -- choose ones we need
        extras['augmentations'] = dict()
        for k, v in augmentations.items():
            if k in [str(i) for i in train_imgids]:
                extras['augmentations'][int(k)] = v

        extras['augs_x_image'] = [augs_x_image[i] for i in train_imgids]

        assert len(extras['augmentations']) == len(extras['augs_x_image']) \
            == len(train_imgids) == len(train_image_paths)

    else:
        if img_augment:
            extras['total_len'] = len(train_image_paths)
            extras['total_len'] += len(train_image_paths) * nr_augs
            extras['nr_augs'] = nr_augs


    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)

    # iu.simple_writing(val_image_paths, val_image_captions, 'VAL', info, word_map)
    # iu.simple_writing(test_image_paths, test_image_captions, 'TEST', info, word_map)
    
    if img_augment == False and text_augment == False:
        iu.simple_writing(train_image_paths, train_image_captions, 'TRAIN', info, word_map)
    elif img_augment == True and text_augment == True:
        print('\nJoint augmentation...')
        iu.joint_writing(train_image_paths, train_image_captions, 'TRAIN', info, \
            word_map, extras, image_transform)
    elif img_augment == True and text_augment == False:
        print('\nImage augmentation...')
        iu.imgaug_writing(train_image_paths, train_image_captions, 'TRAIN', info, \
            word_map, extras, image_transform)
    elif img_augment == False and text_augment == True:
        print('\nText augmentation...')
        iu.textaug_writing(train_image_paths, train_image_captions, 'TRAIN', info, \
            word_map, extras)
    else:
        print('Train input files could not be created. Enter proper conditions.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('--dataset', '-d', type=str, default='vizwiz', \
        required=False, help='Which dataset should be prepared?')
    parser.add_argument('--image_folder', '-f', type=str, default=P.base_vizwiz, \
        required=False, help='Directory of images & annotations.')
    parser.add_argument('--output_folder', '-o', type=str, \
        default=P.base_vizwiz + 'output_files_v3/', required=False, \
        help='Output files directory.')
    
    parser.add_argument('--image_augment', '-i', type=bool, default=False, \
        required=False, help='Generate more images?')
    parser.add_argument('--caption_augment', '-c', type=bool, default=False, \
        required=False, help='Generate more captions?')
    args = parser.parse_args()
    
    
    # Create input files (along with updated word map)
    create_input_files(dataset=args.dataset,
                       image_folder=args.image_folder, # P.base_vizwiz,
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder=args.output_folder,# P.base_vizwiz + 'output_files_v2/',
                       wordmap=P.word_map,
                       max_len=50,
                       text_augment=args.caption_augment,
                       img_augment=args.image_augment)
