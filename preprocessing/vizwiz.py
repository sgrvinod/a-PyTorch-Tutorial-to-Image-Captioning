import json
import numpy as np
import os
from .paths import base_vizwiz, path2model
from transformers import BertTokenizer
from tqdm import tqdm

tk = BertTokenizer.from_pretrained(path2model)

class CocoFlickr():
    def __init__(self, karpathy_coco, max_len, image_folder):
        self.karpathy_coco = karpathy_coco
        self.max_len = max_len
        self.image_folder = image_folder
        self.dataset = 'coco'

        data = self.read_data()
    
    def read_data(self):
        # Read Karpathy JSON
        with open(self.karpathy_coco, 'r') as j:
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
                if len(c['tokens']) <= self.max_len:
                    captions.append(c['raw'])  # NOTE

            if len(captions) == 0:
                continue

            path = os.path.join(self.image_folder, img['filepath'], img['filename']) if self.dataset == 'coco' else os.path.join(
                self.image_folder, img['filename'])

            if img['split'] in {'train', 'restval'}:
                train_image_paths.append(path)
                train_image_captions.append(captions)        
            elif img['split'] in {'val'}:
                val_image_paths.append(path)
                val_image_captions.append(captions)
            elif img['split'] in {'test'}:
                test_image_paths.append(path)
                test_image_captions.append(captions)


class VizWiz():
    # NOTE 20220509: support for ids
    def __init__(self, main_path, max_len=50, tokenizer=tk):
        self.main_path = main_path
        self.max_len = max_len
        self.tokenizer = tokenizer
        
        self.train_ids = np.load(f'{main_path}vw_train.npy')
        self.val_ids = np.load(f'{main_path}vw_val.npy')
        self.test_ids = np.load(f'{main_path}vw_test.npy')
    
        train_i, train_a = self.store_img_anns('train')
        test_i, test_a = self.store_img_anns('val')
        
        train_x = self.image_paths_ids(train_i, self.train_ids, 'train')
        train_y = self.captions_with_id(train_a, self.train_ids)
        self.train_paths, self.train_captions = \
            self.list_paths_captions(train_x, train_y, self.train_ids)
        
        val_x = self.image_paths_ids(train_i, self.val_ids, 'train')
        val_y = self.captions_with_id(train_a, self.val_ids)
        self.val_paths, self.val_captions = \
            self.list_paths_captions(val_x, val_y, self.val_ids)

        test_x = self.image_paths_ids(test_i, self.test_ids, 'val')
        test_y = self.captions_with_id(test_a, self.test_ids)
        self.test_paths, self.test_captions = \
            self.list_paths_captions(test_x, test_y, self.test_ids)


    def tokenize(self, s):  # maybe change.
        return self.tokenizer.encode(s)


    def store_img_anns(self, split):
        """
        Reads annotation JSON file and stores image and annotation info.
        """
        with open(f'{self.main_path}annotations/{split}.json', 'r') as j:
            data = json.load(j)
        images = data['images']
        annotations = data['annotations']
        return images, annotations


    def image_paths_ids(self, images, ids, split):
        """
        Stores image paths in a dict with their IDs
        """
        image_paths = dict()
        s_ids = set(ids)

        for image in images:
            imgid = image['id']
            if imgid in s_ids:
                imagepath = f"{self.main_path}{split}/{image['file_name']}"
                image_paths[image['id']] = imagepath
        
        return image_paths

   
    def captions_with_id(self, annotations, ids):
        """
        Stores captions in a dictionary with their respective image ID.
        """
        captions_dict = dict()
        s_ids = set(ids)

        for item in annotations:
            imgid = item['image_id']
            if imgid in s_ids:
                caption_tokens = self.tokenize(item['caption'])
                if imgid not in captions_dict:
                    captions_dict[imgid] = list()
                if len(caption_tokens) <= self.max_len + 2:
                    captions_dict[imgid].append(item['caption'])
        
        return captions_dict


    def list_paths_captions(self, img_dict, cap_dict, ids):
        paths = list()
        captions = list()
        
        for i in ids:
            paths.append(img_dict[i])
            captions.append(cap_dict[i])
        
        return paths, captions

    
    # def train_augmentation(self, captions_ids, target_ids):
    #     """
    #     Has only performed once.
    #     """
        
    #     with open('originals.json', 'w') as outfile:
    #         json.dump(captions_ids, outfile, indent=2)
        
    #     augmented_captions = dict()
    #     augs_x_img = list()

    #     for i in tqdm(target_ids):
    #         ac = text_augmentation(captions_ids[i])
    #         augmented_captions[i] = ac
    #         augs_x_img.append(len(ac))
        
    #     with open('augmentations.json', 'w') as j:
    #         json.dump(augmented_captions, j, indent=2)
        
    #     with open('augsXimage.json', 'w') as j:
    #         json.dump(augs_x_img, j)
