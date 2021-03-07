import sys
import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.patheffects as PathEffects
import matplotlib
from tqdm import tqdm
tqdm.pandas()
import skimage.io as io
from download import *
from caption import *
from helper import *
print(os.getcwd())
sys.path.insert(0, 'src/data/cocoapi/PythonAPI')
from pycocotools.coco import COCO
# from pycocotools import mask as maskUtils
class ImageCaptionExplainer(object):
    def __init__(self, img_dir, out_dir, model_fp, wordmap_fp, beam_size, coco_fp, caption_model_id=None, wordmap_id=None):
        self.img_dir = img_dir
        self.out_dir = out_dir
        self.model_fp = model_fp
        self.wordmap_fp = wordmap_fp
        if caption_model_id is not None or wordmap_id is not None:
            self.caption_model_id = caption_model_id
            self.wordmap_id = wordmap_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.coco = COCO(coco_fp)
        self.raw_imgs = self._get_raw_imgs()
        self.img_ids = [int(i.split('/')[-2].split('_')[-1]) for i in self.raw_imgs]
#         self.counterfactuals = self._get_counterfactual_imgs()

        self.encoder, self.decoder = self._load_model()

        with open(self.wordmap_fp, 'r') as j:
            word_map = json.load(j)
        self.word_map = word_map
        self.rev_word_map = {v: k for k, v in self.word_map.items()}  # ix2word

        self.beam_size = beam_size
        self.alphas, self.seqs = self._get_alphas_seqs()
#         self.seqs = {}
        self.ids = self._get_img_ann_ids()


#     # Visualize caption and attention of best sequence
# #     visualize_att(img_fp, seq, alphas, rev_word_map, fname, smooth)
#     if visualize == True: visualize_att(img_fp, seq, alphas, rev_word_map, smooth)
#     return ' '.join(words[1:-1])

    def __repr__(self):
        return f'ImageCaptionExplainer(img_dir=\'{self.img_dir}\', out_dir=\'{self.out_dir}\')'

    def _get_img_ann_ids(self):
        ids = {}
#         ann_ids = []
        for idx, img in enumerate(self.img_ids):
            ids[img] = []
            load_fp = f'{self.img_dir}/{img}/counterfactuals'
            # get raw alphas, seq
            # get_viz(wordmap_fp, in_fp, beam_size, out_fp, smooth, visualize=False)
            for i, ann in enumerate(os.listdir(load_fp)):
                # get counterfactual captions
                if 'raw' in ann or 'auto' in ann: continue
                ann_id = int(ann.split('_')[-1].strip('.png'))
#                 ann_ids.append(ann_id)
                ann_fp = load_fp + ann
                ids[img].append(ann_id)
            ids[img] = list(set(ids[img]))
#                 self.alphas[ann_id], self.seqs[ann_id] = caption_image_beam_search(self.encoder, self.decoder, ann, self.word_map, self.beam_size)
#         self.ann_ids = ann_ids
#         ids
        return ids


    def _get_alphas_seqs(self):
#         ann_ids = []
        #  Encode, decode with attention and beam search
        seqs = {}
        alphas = {}
        for idx, img in enumerate(self.img_ids):
            load_fp = f'{self.img_dir}/{img}/'
            # get raw alphas, seq
            try:
                img_fp = load_fp + 'raw.png'
                seqs[img], alphas[img] = caption_image_beam_search(self.encoder, self.decoder, img_fp, self.word_map, self.beam_size)

            except FileNotFoundError:
                img_fp = load_fp + f'raw_{img}.png'
                seqs[img], alphas[img] = caption_image_beam_search(self.encoder, self.decoder, img_fp, self.word_map, self.beam_size)
            # get_viz(wordmap_fp, in_fp, beam_size, out_fp, smooth, visualize=False)
            for i, ann in enumerate(os.listdir(f'{self.img_dir}/{img}/counterfactuals')):
                # get counterfactual captions
                if 'raw' in ann or 'maps' in ann or 'auto' in ann: continue
                ann_id = int(ann.split('_')[-1].strip('.png'))
                ann_fp = f'{self.img_dir}/{img}/counterfactuals/{ann}'
                seqs[ann_id], alphas[ann_id] = caption_image_beam_search(self.encoder, self.decoder, ann_fp, self.word_map, self.beam_size)
#         self.ann_ids = ann_ids
        return alphas, seqs

    def word_alpha_tuple(self, img_id):
#         word_alpha = {}
        words = [self.rev_word_map[ind] for ind in self.seqs[img_id][1:-1]]
        alphas = self.alphas[img_id][1:-1]
        print(f'word, alpha length: {(len(words), len(alphas))}')
        assert len(words) == len(alphas), f'{len(words)} != {len(alphas)}'
        return tuple(zip(words, alphas))

    def _get_caption(self, img_id):
#         print(img_fp)
#         img_id = int(img.split('/')[-2])
        words = [self.rev_word_map[ind] for ind in self.seqs[img_id]]
        caption = ' '.join(words[1:-1])
#         self.seq = ' '.join(words[1:-1])
        return caption

    def _load_model(self):
        # Load model
        try: checkpoint = torch.load(self.model_fp, map_location=str(self.device))
        except FileNotFoundError:
            print(f'Downloading caption model to {self.model_fp}...')
            download_file_from_google_drive(self.caption_model_id, self.model_fp)
            print(f'Downloading wordmap to {self.wordmap_fp}...')
            download_file_from_google_drive(self.wordmap_id, self.wordmap_fp)
            checkpoint = torch.load(self.model_fp, map_location=str(self.device))
        decoder = checkpoint['decoder']
        decoder = decoder.to(self.device)
        decoder.eval()
        encoder = checkpoint['encoder']
        encoder = encoder.to(self.device)
        encoder.eval();
        return encoder, decoder

    def _get_raw_imgs(self):
        raw_imgs = []
#         caption_dict = {}
        for img_folder in (os.listdir(self.img_dir)):
            load_fp = f'{self.img_dir}/{img_folder}'
            for file in (os.listdir(f'{self.img_dir}/{img_folder}')):
                in_fp = f'{load_fp}/{file}'
                if 'raw' in file:
                    raw_imgs.append(in_fp)
        return raw_imgs


    def _get_img_caption(self, img_id, img_caption_dict, ann_caption_dict):
        load_fp = f'{self.img_dir}/{img_id}/counterfactuals'
        # get raw img captions
        img_caption_dict[img_id] =  self._get_caption(img_id)
        # get_viz(wordmap_fp, in_fp, beam_size, out_fp, smooth, visualize=False)
        for i, ann in enumerate(os.listdir(load_fp)):
            # get counterfactual captions
            if 'raw' in ann or 'auto' in ann: continue
            ann_id = int(ann.split('_')[-1].strip('.png'))
#                 out_fp+=f'maps_{ann_id}.png'
            ann_fp = load_fp + ann
            words = self._get_caption(ann_id)
            ann_caption_dict[(img_id, ann_id)] = [words]
        return img_caption_dict, ann_caption_dict

    def _get_captions(self, img_id=None):
        img_caption_dict = {}
        ann_caption_dict = {}
        if img_id is not None:
            return self._get_img_caption(img_id, img_caption_dict, ann_caption_dict)

        for idx, img in enumerate(self.img_ids):
            img_caption_dict, ann_caption_dict = self._get_img_caption(img, img_caption_dict, ann_caption_dict)
        return img_caption_dict, ann_caption_dict

    def get_bert_similarities(self, img_id=None):
        raw_caption_dict, ann_caption_dict = self._get_captions(img_id)
#         ann_caption_dict = self._get_captions(self.out_dir, self.out_dir)

        ann_df = pd.DataFrame(ann_caption_dict).transpose().reset_index().rename(columns={0:'ann_caption'})
        caption_df = pd.Series(raw_caption_dict).to_frame().rename(columns={0:'raw_caption'}).merge(ann_df, left_index=True, right_on='level_0')
        caption_df = caption_df.rename(columns={'level_0':'img_id', 'level_1':'ann_id'})
#         print(caption_df)
        caption_df.loc[(caption_df['raw_caption'] == caption_df['ann_caption']), 'dist_from_raw'] = 0
        if caption_df['dist_from_raw'].isnull().sum() > 0:
            caption_df.loc[list(caption_df['dist_from_raw'].isnull().index), 'dist_from_raw'] = \
                caption_df.loc[(caption_df['dist_from_raw'].isnull())].progress_apply(lambda x: compare_two_sentences(x['raw_caption'], x['ann_caption']), axis=1)
            caption_df['dist_from_raw'] = caption_df['dist_from_raw'].fillna(0)
        self.caption_similarities = caption_df
        return caption_df


    def _get_mapping(self, x, low_val, high_val):
        if (high_val - low_val)== 0: return 0, 0, 0
        green_val = (x - low_val) / (high_val - low_val)
        red_val = 1 - green_val
    #     red_val = ((x - high_val) / (high_val - low_val)) + 1
        return red_val, green_val, 0

    def explain_image(self, img_id=None, methods=['bert']):
        assert img_id is not None, 'running explain_image without img_id not supported yet, please supply an img_id'

        def generate_polygon(img_caption_df, ann, seg, vals, x_y, polygons):
            val = img_caption_df.loc[img_caption_df['ann_id']==ann['id'], 'dist_from_raw'].values[0]
            vals.append(val)
            poly = np.array(seg).reshape((int(len(seg)/2), 2))
            poly = Polygon(poly, label=str(val))
            x_y.append((np.mean(poly.get_xy()[:, 0]), np.mean(poly.get_xy()[:, 1])))
            polygons.append(poly)

        for method in methods:
            if method == 'bert': img_caption_df = self.get_bert_similarities(img_id) # in the future additional methods will be explored, like 'alphas'

            img = self.coco.loadImgs([img_id])[0]
            I = io.imread(img['coco_url'])
            plt.imshow(I); plt.axis('off')

            ax = plt.gca()
            ax.set_autoscale_on(False)

            polygons = []
            vals = []
            # x_y = []
            anns = sorted(self.ids[img_id])
            ann_lst = self.coco.loadAnns(anns)
            # print(f'num anns: {(ann_lst[1]["bbox"])}')
            x_y = {}
            for ann in ann_lst:
            #     c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
                # print(ann['id'])
                # try:
                #     print(ann['segmentation'][0])
                # except KeyError:
                #     print(img_id)
                #     print(ann['segmentation'])
                if type(ann['segmentation']) ==  list:
                    # poly_lst = []
                    for seg in ann['segmentation']:
                        val = img_caption_df.loc[img_caption_df['ann_id']==ann['id'], 'dist_from_raw'].values[0]
                        vals.append(val)
                        poly = np.array(seg).reshape((int(len(seg)/2), 2))
                        # poly_lst.append(poly.get_xy())
                        poly = Polygon(poly, label=str(val))
                        # x_y.append((np.median(poly.get_xy()[:, 0]), np.median(poly.get_xy()[:, 1])))
                        polygons.append(poly)
                        # poly_lst = np.array(poly_lst)
                        # print(f'poly shape: {np.array([poly]).shape}')
                    # print(f'shape: {poly_lst.shape}')
                    # poly_lst = np.concatenate(poly_lst, axis=0)
                    for i in range(len(vals)):
                        # x_y[i] = x_y.get(i,(np.median(poly_lst[:, 0]), np.median(poly_lst[:, 1])))
                        x, y, width, height = tuple(ann['bbox'])
                        x_y[i] = x_y.get(i, (x+width/2, y+height/2))

                    # print(f'new shape: {poly_lst.shape}')

                    # c = self._get_mapping(val, )
                else:
                    continue # we are skipping annotation that are potentially crowds for now
            # max_val = max(vals)
            # green_vals = np.array(vals)
            # green_vals /= max_val  # this will be the green content in each annotation
            # red_vals = (1 - green_vals)
            # blue_vals = np.zeros(len(red_vals))
            # color = list(zip(red_vals, green_vals, blue_vals))
            min_val, max_val = min(vals), max(vals)
            norm = matplotlib.colors.Normalize(vmin=min_val, vmax=max_val)
            cmap = 'RdYlGn'
            scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)

            # print(f'vals: {vals}')
            color = [scalar_map.to_rgba(i) for i in vals]
#             color = [self._get_mapping(i, min_val, max_val) for i in vals]
            # print(x_y)
            for i, c in enumerate(color):
                # print(f'color: {c}')
                centerx, centery = x_y[i]
                text = ax.text(centerx, centery, str(round(vals[i], 4)), size=8, color=c)
                outline='black'
#                 if c == (0, 0, 0, 1): # color is black, set text outline to white
#                     outline = 'white'
#                 else:
#                     outline = 'black'
                text.set_path_effects([PathEffects.withStroke(linewidth=.6,foreground=outline)])
            p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.3)
            ax.add_collection(p)
            p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=1.7)
            ax.add_collection(p);

            plt.colorbar(mappable=scalar_map, ax=ax)
            out_fp = f'{self.out_dir}/{img_id}'
            dir_creator(out_fp)
            out_fp += f'/{method}_object_importance_{img_id}.png'
            # print(f'current directory: {os.getcwd()}')
            plt.savefig(out_fp, dpi=900);
            plt.close();
            print(f'Wrote to {out_fp}')
