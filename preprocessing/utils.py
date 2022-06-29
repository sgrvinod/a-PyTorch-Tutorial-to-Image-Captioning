import cv2
import h5py
import json
from math import ceil
import nltk
import os
from random import seed, choice, sample
from tqdm import tqdm

import numpy as np

from paths import path2model
from transformers import BertTokenizer

# aliciaviernes: for use of Bert Tokenizer, the quickest solution
# is to download the model locally and enter the local path.
# Function is below.
tokenizer = BertTokenizer.from_pretrained(path2model)


def ensure_five(imcaps, caps_per_image=5):
    if len(imcaps) < caps_per_image:
        caps = imcaps + [choice(imcaps) for _ in range(caps_per_image - len(imcaps))]
    else:
        caps = sample(imcaps, k=caps_per_image)  # NOTE sampling captions after all
    return caps


def image_preprocessing(img):

    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = cv2.resize(img, (256, 256))  # aliciaviernes modification
    img = img.transpose(2, 0, 1)

    assert img.shape == (3, 256, 256)
    assert np.max(img) <= 255

    return img


def caption_encoding(c, max_len):

    global tokenizer

    # New function - aliciaviernes - including BERT tokenizer
    encoding = tokenizer.encode(c)

    if len(encoding) > max_len:
        encoding = encoding[:max_len - 1]
        encoding.append(102)

    padded_encoding = encoding.copy()

    while len(padded_encoding) < max_len:
        padded_encoding.append(0)

    return padded_encoding, len(encoding) - 2


def old_encoding(c, max_len, word_map):
    c = nltk.word_tokenize(c.lower())
    if len(c) > 50:
        c = c[:50]
    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))
    # Find caption lengths
    c_len = len(c) + 2
    return enc_c, c_len

# train_image_paths, train_image_captions, 'TRAIN'

def simple_writing(impaths, imcaps, split, info, word_map):

        # Create output file for images
        with h5py.File(os.path.join(info['output_folder'], \
            split + '_IMAGES_' + info['base_filename'] + '.hdf5'), 'a') as h:
            
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = info['captions_per_image']

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            # all captions and all caplens in a list.
            enc_captions, caplens = list(), list()

            # need to fix the other thing
            for i, path in enumerate(tqdm(impaths)):
                # NOTE //: aufrunden, %: abrunden
                # Sample captions
                captions = ensure_five(imcaps=imcaps[i], caps_per_image=info['captions_per_image'])
                # Sanity check NOTE to remove potentially
                assert len(captions) == info['captions_per_image']
                
                # Read images
                img = cv2.imread(impaths[i])
                img = image_preprocessing(img)
                images[i] = img

                for j, c in enumerate(captions):
                    # Encode captions
                    enc_c, c_len = old_encoding(c, info['max_len'], word_map=word_map)
                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Sanity check
            assert images.shape[0] * info['captions_per_image'] \
                == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(info['output_folder'], \
                split + '_CAPTIONS_' + info['base_filename'] + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(info['output_folder'], \
                split + '_CAPLENS_' + info['base_filename'] + '.json'), 'w') as j:
                json.dump(caplens, j)


def joint_writing(impaths, imcaps, split, info, word_map, extras, img_func):

    # all images + extra for augmentations
    images = dict()
    # all captions and all caplens in a list.
    enc_captions, caplens = list(), list()
    # index for image file writing
    index = 0

    augs_X_image = extras['augs_x_image']
    ids = extras['train_imgids']

    extra_images = list()
    for img in augs_X_image:
        extra_images.append(ceil(img/ 5))

    print("\nReading %s images and captions...\n" % split)    

    for i, n in enumerate(tqdm(ids)):  # in enumerate(tqdm(ids))
        
        # Caption writing
        captions = ensure_five(imcaps=imcaps[i], caps_per_image=info['captions_per_image'])
        
        # Split all augmentations into smaller chunks
        chunks = [extras['augmentations'][n][x:x+info['captions_per_image']] \
                for x in range(0, len(extras['augmentations'][n]), \
                info['captions_per_image'])]
        
        # is the last chunk big enough?
        chunks[-1] = ensure_five(chunks[-1], caps_per_image=info['captions_per_image'])
        for chunk in chunks:
            assert len(chunk) == info['captions_per_image']
        
        for j, c in enumerate(captions):
            # Encode captions
            enc_c, c_len = old_encoding(c, info['max_len'], word_map=word_map)
            # print('Caption', c)
            # print('Encoding', enc_c)
            # print('Encoding length', len(enc_c))
            # print()
            enc_captions.append(enc_c)
            caplens.append(c_len)
            
        for chunk in chunks:
            for j, c in enumerate(chunk):
                enc_c, c_len = old_encoding(c, info['max_len'], word_map=word_map)
                # print('* Augmented caption:', c)
                # print('Encoding', enc_c)
                # print('Encoding length', len(enc_c))
                # print()
                enc_captions.append(enc_c)
                caplens.append(c_len)
        
        # Read images
        img = cv2.imread(impaths[i])
        # Save image to HDF5 file
        img = image_preprocessing(img)
        images[index] = img
        index += 1
        
        for i in range(extra_images[i]):
            new_img = img_func(impaths[i], save=False)
            images[index] = image_preprocessing(new_img)
            index += 1
    
    assert len(images) * info['captions_per_image'] == len(enc_captions) == len(caplens)

    print('\nWriting caption files...')
    # Save encoded captions and their lengths to JSON files
    with open(os.path.join(info['output_folder'], split + '_CAPTIONS_' + info['base_filename'] + '.json'), 'w') as j:
        json.dump(enc_captions, j)

    with open(os.path.join(info['output_folder'], split + '_CAPLENS_' + info['base_filename'] + '.json'), 'w') as j:
        json.dump(caplens, j)

    # write images
    print('\nWriting image file...')

    with h5py.File(os.path.join(info['output_folder'], split + '_IMAGES_' + info['base_filename'] + '.hdf5'), 'a') as h:
        
        h.attrs['captions_per_image'] = info['captions_per_image']
        imgs = h.create_dataset('images', (len(images), 3, 256, 256), dtype='uint8')
        for k, v in images.items():
            imgs[k] = v


def imgaug_writing(impaths, imcaps, split, info, word_map, extras, img_func):
    
    with h5py.File(os.path.join(info['output_folder'], split + '_IMAGES_' + info['base_filename'] + '.hdf5'), 'a') as h:
            
        # Make a note of the number of captions we are sampling per image
        h.attrs['captions_per_image'] = info['captions_per_image']

        # Create dataset inside HDF5 file to store images
        images = h.create_dataset('images', \
            (extras['total_len'], 3, 256, 256), dtype='uint8')

        print("\nReading %s images and captions, storing to file...\n" % split)

        # all captions and all caplens in a list.
        enc_captions, caplens = list(), list()

        # index for image file writing
        index = 0

        # for each image...
        for i, path in enumerate(tqdm(impaths)):
            
            # Sample captions
            captions = ensure_five(imcaps=imcaps[i], caps_per_image=info['captions_per_image'])
            
            # Read images
            img = cv2.imread(impaths[i])
            # Image augmentation & Save image to HDF5 file
            images[index] = image_preprocessing(img)
            
            for j in range(extras['nr_augs']):
                index += 1
                new_img = img_func(impaths[i], save=False)
                images[index] = image_preprocessing(new_img)
            
            for j, c in enumerate(captions):
                # Encode captions
                enc_c, c_len = old_encoding(c, info['max_len'], word_map=word_map)

                for x in range(extras['nr_augs'] + 1):  # considering original + augmentations
                    enc_captions.append(enc_c)
                    caplens.append(c_len)

        # Sanity check
        assert images.shape[0] * info['captions_per_image'] == len(enc_captions) == len(caplens)

        # Save encoded captions and their lengths to JSON files
        with open(os.path.join(info['output_folder'], split + '_CAPTIONS_' + info['base_filename'] + '.json'), 'w') as j:
            json.dump(enc_captions, j)

        with open(os.path.join(info['output_folder'], split + '_CAPLENS_' + info['base_filename'] + '.json'), 'w') as j:
            json.dump(caplens, j)


def textaug_writing(impaths, imcaps, split, info, word_map, extras):

    # all images + extra for augmentations
    images = dict()
    # all captions and all caplens in a list.
    enc_captions, caplens = list(), list()
    # index for image file writing
    index = 0

    augs_X_image = extras['augs_x_image']
    ids = extras['train_imgids']

    extra_images = list()
    for img in augs_X_image:
        extra_images.append(ceil(img/ 5))

    print("\nReading %s images and captions...\n" % split)    

    for i, n in enumerate(tqdm(ids)):  # in enumerate(tqdm(ids))
        
        # Caption writing
        captions = ensure_five(imcaps=imcaps[i], caps_per_image=info['captions_per_image'])
        
        # Split all augmentations into smaller chunks
        chunks = [extras['augmentations'][n][x:x+info['captions_per_image']] \
                for x in range(0, len(extras['augmentations'][n]), \
                info['captions_per_image'])]
        
        # is the last chunk big enough?
        chunks[-1] = ensure_five(chunks[-1], caps_per_image=info['captions_per_image'])
        for chunk in chunks:
            assert len(chunk) == info['captions_per_image']
        
        for j, c in enumerate(captions):
            # Encode captions
            enc_c, c_len = old_encoding(c, info['max_len'], word_map=word_map)
            # print('Caption', c)
            # print('Encoding', enc_c)
            # print('Encoding length', len(enc_c))
            # print()
            enc_captions.append(enc_c)
            caplens.append(c_len)
            
        for chunk in chunks:
            for j, c in enumerate(chunk):
                enc_c, c_len = old_encoding(c, info['max_len'], word_map=word_map)
                # print('* Augmented caption:', c)
                # print('Encoding', enc_c)
                # print('Encoding length', len(enc_c))
                # print()
                enc_captions.append(enc_c)
                caplens.append(c_len)
        
        # Read images
        img = cv2.imread(impaths[i])
        # Save image to HDF5 file
        img = image_preprocessing(img)
        images[index] = img
        index += 1
        
        for i in range(extra_images[i]):
            images[index] = img
            index += 1
    
    assert len(images) * info['captions_per_image'] == len(enc_captions) == len(caplens)

    print('\nWriting caption files...')
    # Save encoded captions and their lengths to JSON files
    with open(os.path.join(info['output_folder'], split + '_CAPTIONS_' + info['base_filename'] + '.json'), 'w') as j:
        json.dump(enc_captions, j)

    with open(os.path.join(info['output_folder'], split + '_CAPLENS_' + info['base_filename'] + '.json'), 'w') as j:
        json.dump(caplens, j)

    # write images
    print('\nWriting image file...')
    with h5py.File(os.path.join(info['output_folder'], split + '_IMAGES_' + info['base_filename'] + '.hdf5'), 'a') as h:
        
        h.attrs['captions_per_image'] = info['captions_per_image']
        
        imgs = h.create_dataset('images', (len(images), 3, 256, 256), dtype='uint8')
        
        for k, v in images.items():
            imgs[k] = v
