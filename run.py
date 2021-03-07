import json
import sys
# import torch
# from src.utils import create_input_files
# from src.train import main
# from src.eval import evaluate
# from src.caption import caption_image_beam_search, visualize_att
from src.counterfactuals import create_mask_input, generate_counterfactuals
from src.img_caption_explainer import ImageCaptionExplainer

def data(test=False):
    root = ''
    if test:
        f = open(root+"config/create_input_files_test.json")
    else:
        f = open(root+"config/create_input_files.json")
    jsonread = json.load(f)
    create_input_files(dataset=jsonread['dataset'],
                       karpathy_json_path=root+jsonread['karpathy_json_path'],
                       image_folder=jsonread['image_folder'],
                       captions_per_image=jsonread['captions_per_image'],
                       min_word_freq=jsonread['min_word_freq'],
                       output_folder=root+jsonread['output_folder'],
                       max_len=jsonread['max_len'],)

def train(test=False):
    main(test)

def evaluate_model(test=False):
    if test:
        f = open('config/eval_test.json')
    else:
        f = open('config/eval.json')
    jsonread = json.load(f)
    beam_size = jsonread['beam_size']
    print("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_size, evaluate(beam_size, test)))


def generate_viz(test=False):
    if test:
        f = open('config/caption_test.json')
    else:
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

def counterfactual_production(test=False, year=None):
    if test:
        fname = 'config/counterfactual_test.json'
    else:
        fname = 'config/counterfactual.json'
    if year == '2014':
        fname = 'config/counterfactual_2014.json'

    with open(fname, 'r') as f:
        jsonread = json.load(f)
    img_dir = jsonread['img_dir']
    temp_dir = jsonread['temp_dir']
    out_dir = jsonread['out_dir']
    annotation_fp = jsonread['annotation_fp']
    checkpoint_dir = jsonread['checkpoint_dir']
    model_id = jsonread['model_id']

    create_mask_input(img_dir, temp_dir, out_dir, annotation_fp, False)
    generate_counterfactuals(temp_dir, checkpoint_dir, model_id)

def explain_model(test=False):
    if test:
        fname = 'config/counterfactual_test.json'
    else:
        fname = 'config/counterfactual.json'

    with open(fname, 'r') as f:
        jsonread=json.load(f)

    temp_dir=jsonread['temp_dir']
    out_dir=jsonread['out_dir']
    model_fp=jsonread['model_fp']
    wordmap_fp=jsonread['wordmap_fp']
    coco_fp=jsonread['annotation_fp']
    beam_size=jsonread['beam_size']
    caption_model_id=jsonread['caption_model_id']
    wordmap_id = jsonread['wordmap_id']

    img_caption = ImageCaptionExplainer(img_dir=temp_dir, out_dir=out_dir, \
    model_fp=model_fp, wordmap_fp=wordmap_fp, coco_fp=coco_fp, beam_size=beam_size,\
    caption_model_id=caption_model_id, wordmap_id=wordmap_id)

    for img_id in img_caption.ids:
        img_caption.explain_image(img_id)


def all():
    data()
    train()
    evaluate_model()
    generate_viz()

def coco2014():
    counterfactual_production(year='2014')

def test():
    # data(True)
    # train(True)
    # evaluate_model(True)
    counterfactual_production(test=True)
    explain_model(test=True)
    # generate_viz(True)

if __name__ == '__main__':
    # try:
    globals()[sys.argv[1]]()
    # except IndexError:
    #     data()
    #     train()
    #     evaluate_model()
    #     generate_viz()
