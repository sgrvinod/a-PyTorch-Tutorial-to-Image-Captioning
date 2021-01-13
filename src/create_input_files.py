from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='coco',
                       karpathy_json_path='data/raw/anno2014/dataset_coco.json',
                       image_folder='data/raw/',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='data/raw/',
                       max_len=50)

#     image folder and output folder need to be the same??