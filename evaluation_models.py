# import torch
import csv
import os
import paths as P
from eval import *


datasets = {
    'coco': f'{P.base_coco}original_output_files/',
    'vizwiz': f'{P.base_vizwiz}output_files_v3/',
}

# RESNEXT EVALUATION
checkpoints_aug = {
    './checkpoints/aug_nococo/BEST_checkpoint_vizwiz_good_quality_decoder_finetuning.pth.tar', 
    './checkpoints/aug_nococo/BEST_checkpoint_vizwiz_textaug_decoder_encoder_finetuning.pth.tar',  
    './checkpoints/aug_nococo/BEST_checkpoint_vizwiz_textaug_decoder_finetuning.pth.tar', 
    './checkpoints/aug_nococo/BEST_checkpoint_vizwiz_imgaug_decoder_finetuning.pth.tar', 
    './checkpoints/aug_nococo/BEST_checkpoint_vizwiz_jointaug_decoder_finetuning.pth.tar',
    './checkpoints/aug_nococo/BEST_checkpoint_vizwiz_good_quality_decoder_encoder_finetuning.pth.tar',  
}

word_map_file = f'./WORDMAP_vizwiz_5_cap_per_img_5_min_word_freq.json'
# word_map_file = '/home/aanagnostopoulou/DATA/coco/original_output_files/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'

print('*' * 100)

results = [['Dataset', 'Checkpoint', 'BLEU-4']]

for checkpoint in checkpoints_aug:

    for dataset in datasets:

        data_name = f'{dataset}_5_cap_per_img_5_min_word_freq'
        data_folder = datasets[dataset]

        result = evaluate(data_name, data_folder, checkpoint, word_map_file)
        results.append([dataset, checkpoint, result])
        print('*' * 100)


with open("results.csv", "w", newline="") as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(results)
