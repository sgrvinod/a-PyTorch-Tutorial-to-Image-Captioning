import json
import sys
import torch
from src.utils import create_input_files
from src.train import main
from src.eval import evaluate
from src.caption import caption_image_beam_search, visualize_att

def data():
    root = ''
    f = open(root+"config/create_input_files.json")
    jsonread = json.load(f) 
    create_input_files(dataset=jsonread['dataset'],
                       karpathy_json_path=root+jsonread['karpathy_json_path'],
                       image_folder=jsonread['image_folder'],
                       captions_per_image=jsonread['captions_per_image'],
                       min_word_freq=jsonread['min_word_freq'],
                       output_folder=root+jsonread['output_folder'],
                       max_len=jsonread['max_len'],)
    
def train():
    main()
    
def evaluate_model():
    f = open('config/eval.json')
    jsonread = json.load(f)
    beam_size = jsonread['beam_size']
    print("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_size, evaluate(beam_size)))
    
def generate_viz():
    f = open('config/caption.json')
    jsonread = json.load(f) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    checkpoint = torch.load(jsonread['model_fp'], map_location=str(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map (word2ix)
    with open(jsonread['wordmap_fp'], 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    # Encode, decode with attention and beam search
    seq, alphas = caption_image_beam_search(encoder, decoder, jsonread['img_fp'], word_map, jsonread['beam_size'])
    alphas = torch.FloatTensor(alphas)

    # Visualize caption and attention of best sequence
    visualize_att(jsonread['img_fp'], seq, alphas, rev_word_map, jsonread['smooth'])
    
def all():
    data()
    train()
    evaluate_model()
    generate_viz()
    
# def test():
    

if __name__ == '__main__':
    try:
        globals()[sys.argv[1]]()
    except:
        data()
        train()
        evaluate_model()
        generate_viz()