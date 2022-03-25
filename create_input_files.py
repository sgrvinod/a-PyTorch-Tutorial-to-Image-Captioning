from utils import create_input_files
import paths as P

if __name__ == '__main__':
    
    # Create input files (along with word map)
    create_input_files(dataset='vizwiz',
                       karpathy_json_path=P.karpathy_coco,
                       image_folder=P.image_vizwiz,
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder=P.output_vizwiz,
                       max_len=50,
                       text_augment=False,
                       img_augment=True)
