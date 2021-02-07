The purpose of this code is generating captions from an image and creating attention maps to help explain the model’s reasoning for the captions generated. Work is currently in progress to integrate an image perturbation model to experiment on how captions change when objects are removed from an image.

This project has fou targets: data; train; evaluate_model; generate_viz’ . The data target loads in the COCO dataset and prepares it for our image captioning model. The train target builds the encoder and decoder in our image captioning model and trains it with the COCO dataset. The evaluate_model target evaluates the trained model using beam search caption generation and BLEU score. The generate_viz target generates a visualization of the attention maps at each stage of the caption generation process.

To run the four targets, clone our repo to the dsmlp server and execute the four command lines: 'python run.py data'; 'python run.py train'; 'python run.py evaluate_model’; ‘python run.py generate_viz’ or ‘python run.py all’ to run all the targets in sequence. To run on a small set of test data execute: ‘python run.py test’. The output images will be saved to the same directory as run.py.


Docker Repo: https://hub.docker.com/layers/136526411/afosado/capstone_project/test_targets/images/sha256-45b91267629b6022b35fe70f5cae48c109b40e967420291e77ade205887b44ae?context=explore

