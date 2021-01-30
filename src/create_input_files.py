from utils import create_input_files
import json

if __name__ == '__main__':
    # Create input files (along with word map)
    root = '../'
    f = open(root+"config/create_input_files.json")
    jsonread = json.load(f) 
    create_input_files(dataset=jsonread['dataset'],
                       karpathy_json_path=root+jsonread['karpathy_json_path'],
                       image_folder=jsonread['image_folder'],
                       captions_per_image=jsonread['captions_per_image'],
                       min_word_freq=jsonread['min_word_freq'],
                       output_folder=root+jsonread['output_folder'],
                       max_len=jsonread['max_len'],)