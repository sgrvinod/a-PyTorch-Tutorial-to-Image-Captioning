from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='coco',
                       karpathy_json_path='/datasets/COCO-2017/anno2017/captions_val2017.json',
                       image_folder='/datasets/COCO-2017/val2017/',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='output_images/',
                       max_len=50)

#     "data_path": "/datasets/COCO-2017/val2017/",
#     "json_path": "/datasets/COCO-2017/anno2017/instances_val2017.json",
#     "root": "src/data/out/",
#     "df_path": "src/data/out/cropped_images_df.csv"