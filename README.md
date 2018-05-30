This is a **PyTorch tutorial to Image Captioning**.

This is the first of a series of tutorials I plan to write about _implementing_ cool models on your own with the amazing PyTorch library.

Basic knowledge of PyTorch, convolutional and recurrent neural networks, loss functions is assumed.

Questions, suggestions, or corrections can be posted as issues.

# Contents

[***Objective***](https://github.com/sgrvinod/caption#objective)

[***Concepts***](https://github.com/sgrvinod/caption#concepts) 

[***Overview***](https://github.com/sgrvinod/caption#overview) 

[***Implementation***](https://github.com/sgrvinod/caption#implementation)

[***Frequently Asked Questions***](https://github.com/sgrvinod/caption#faqs)

# Objective

**To build a model that can generate a descriptive caption for an image we provide it.**

In the interest of keeping things simple, let's choose to implement the [_Show, Attend, and Tell_](https://arxiv.org/abs/1502.03044) paper. This is by no means the current state-of-the-art, but is still pretty darn amazing.

This model learns _where_ to look. 

As you generate a caption, word by word, you can see the the model's gaze shifting across the image. 

This is possible because of its _Attention_ mechanism, which allows it to focus on the part of the image most relevant to the word it is going to utter next. 

Here are some captions generated on _test_ images not seen during training or validation:

---

![](./img/plane.png)

---

![](./img/boats.png)

---

![](./img/bikefence.png)

---

![](./img/sheep.png)

---

![](./img/babycake.png)

---

![](./img/dogtie.png)

---

There are more examples at the [end of the tutorial]().

---

# Concepts

* **Image captioning**. You will learn about the general structure of captioning models, how they work, and their implementation.

* **Encoder-Decoder architecture**. Any model that generates sequences will use an Encoder to encode the input into a fixed form and a Decoder to decode it, word by word, into a sequence.

* **Attention**. The use of Attention networks is widespread in deep learning, and with good reason. This is a way for a model to choose only those parts of the encoding that it thinks is relevant to the task at hand. The same mechanism you see employed here can be used in any model where the Encoder's output has multiple points in space or time. In image captioning, you consider some pixels more important than others. In sequence to sequence tasks like machine translation, you consider some words more important than others. 

* **Transfer Learning**. This is when you borrow from an existing model by using parts of it in a new model. This is almost always better than training a new model from scratch (i.e., knowing nothing). As you will see, you can always fine-tune this second-hand knowledge to the specific task at hand. Using pretrained word embeddings is a dumb but valid example. For our image captioning problem, we will use a pretrained Encoder, and then fine-tune it as needed.

* **Beam Search**. This is where you don't let your Decoder be lazy and simply choose the words with the _best_ score at each decode-step. 

# Overview

In this section, I will present a broad overview of this model. I don't really get into the _minutiae_ here - feel free to skip to the implementation section and commented code for details.

### Encoder

The Encoder **encodes the input image with 3 color channels into a smaller image with "learned" channels**.

This smaller encoded image is a summary representation of all that's useful in the original image.

Since we want to encode images, we use Convolutional Neural Networks (CNNs).

We don't need to train an encoder from scratch. Why? Because there are already CNNs trained to represent images. 

For years, people have been building models that are extraordinarily good at classifying an image into one of a thousand categories. It stands to reason that these models capture the essence of an image very well.

I have chosen to use the **101 layered Residual Network trained on the ImageNet classification task**, available in PyTorch. As stated earlier, this is an example of Transfer Learning. You have the option of fine-tuning it to improve performance.

![ResNet Encoder](./img/encoder.png)
<p align="center">
  *ResNet-101 Encoder*
</p>

These models progressively create smaller and smaller representations of the original image, and each subsequent representation is more "learned", with a greater number of channels. The final encoding produced by our ResNet-101 encoder has a size of 14x14 with 4096 channels, i.e., a `4096, 14, 14` size tensor.

I encourage you to experiment with other pre-trained architectures. The paper uses a VGGnet, also pretrained on ImageNet, but without fine-tuning. Either way, modifications are necessary. Since the last layer or two of these models are linear layers coupled with softmax activation for classification, we strip 'em away.

### Decoder

The Decoder's job is to look at the encoded image and generate a caption word by word.

Since it's generating a sequence, it would need to be a Recurrent Neural Network (RNN). We will try the LSTM flavor.

In a typical setting without Attention, you could simply average the encoded image across all pixels. You could then feed this, with or without a linear transformation, into the Decoder as its first hidden state and generate the caption. Each predicted word is used to generate the next word.

![Decoder without Attention](./img/decoder_no_att.png)
<p align="center">
  *Decoding without Attention*
</p>

In a setting _with_ Attention, we want the Decoder to be able to look at different parts of the image at different points in the sequence. For example, while generating the word `football` in `a man holds a football`, the Decoder would know to focus on the - you guessed it - football! 

![Decoding with Attention](./img/decoder_att.png)
<p align="center">
  *Decoding with Attention*
</p>

Instead of the simple average, we use the _weighted_ average across all pixels, with the weights of the important pixels being greater. This weighted representation of the image can be concatenated with the previously generated word at each step to generate the next word.

### Attention

Intuitively, what would you need to estimate the importance of different parts of the image?

You would need to know how much of the sequence you have generated, so you could look at the image and decide what needs describing next. For example, you know that you have mentioned `a man` so far, but you look at the image and notice the aforementioned man is `holding` `a` `football`.

This is exactly what the attention mechanism does - it considers the sequence generated thus far, looks at the image, and _attends_ to the part of it that needs describing next.

![Attention](./img/att.png)
<p align="center">
  *Attention*
</p>

We will use the _soft_ Attention, where the weights of the pixels add up to 1. You could interpret this as finding the probability that a certain pixel is _the_ important part of the image to generate the next word.

(Funny story - when I was a kid growing up in India doing drills at school, the PE teacher would 

### Putting it all together

It might be clear by now what our combined network looks like.

![Putting it all together](./img/model.png)
<p align="center">
  *Encoder + Attention + Decoder*
</p>

- Once the Encoder generates the encoded image, we transform the encoding to create the initial hidden state `h` (and cell state `C`) for the RNN/LSTM Decoder.
- At each decode step,
  - the encoded image and the previous hidden state is used to generate weights for each pixel in the Attention network.
  - the weighted average of the encoding and the previously generated word is fed to the RNN/LSTM Decoder to generate the next word.

### Beam Search

When we generate a word with the Decoder, we are using a linear layer to transform the Decoder's output into a score for each word in the vocabulary.

The straightforward - and quite greedy - option would be to choose the word with the highest score and use it to predict the next word. But this is not optimal because the rest of the sequence hinges on that first word you choose. If that choice isn't the best, everything that follows is sub-optimal. Similarly, the choice of each word in the sequence has consequences for the ones that succeed it.

It might very well happen that if you'd chosen the _third_ best word at that first step, and the _second_ best word at the second step, and so on... _that_ would be the best sequence you could generate.

So, it would be best if we could somehow _not_ decide until we've finished decoding completely, and choose the sequence that has the highest _overall_ score from a basket of candidate sequences.

Beam Search does exactly this:

- At the first decode step, consider the top `k` candidates.
- Generate `k` second words for each of these `k` first words.
- Choose the top `k` [first word, second word] combinations considering additive scores.
- For each of these `k` second words, choose `k` third words, choose the top `k` [first word, second word, third word] combinations.
- Repeat at each decode step.
- After `k` sequences terminate, choose the sequence with the best overall score.

![Beam Search example](./img/beam_search.png)
<p align="center">
  *An illustration of Beam Search with `k`=3.*
</p>

As you can see, some sequences (striked out) may fail early, as they don't make it to the top `k` at the next step. Once `k` sequences (underlined) complete by hitting the `<end>` token, we choose the one with the highest score.

# Implementation

### Dataset

I'm using the MSCOCO '14 Dataset. You need to download the [Training (13GB)](http://images.cocodataset.org/zips/train2014.zip) and [Validation (6GB)](http://images.cocodataset.org/zips/val2014.zip) images.

We will use [Andrej Karpathy's training, validation, and test splits](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip). These also contain the captions, already preprocessed. Note that there are also splits and captions for the Flicker8k and Flicker30k datasets, so feel free to use these instead of MSCOCO if the latter is too large for your computer.

### Inputs to model

Before you read this section, I encourage you to think about this on your own. What inputs does our model need to train?

We will need three.

##### Images

The Encoder encodes these images. Since we're using a pretrained Encoder, we would need to first process the images into the form this pretrained Encoder is used to.

Pretrained ImageNet models available as part of PyTorch's `torchvision` module. [This page](https://pytorch.org/docs/master/torchvision/models.html) details the preprocessing or transform we need to do - pixel values must be in the range [0,1] and we must then normalize the image by the mean and standard deviation of the ImageNet images' RGB channels.

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```
Also, PyTorch follows the NCHW convention, which means the channels dimension (C) must precede the size dimensions.

We will resize all MSCOCO images to 256x256 for uniformity.

To summarize, the images fed to the model must be a `Float` tensor of dimension `N, 3, 256, 256`, and must be normalized by the aforesaid mean and standard deviation. `N` is the batch size.

##### Captions

Captions are both the target and the inputs of the Decoder as each word is used to generate the next word.

To generate the first word, however, we need a *zeroth* word, `<start>`.

At the last word, we should predict `<end>` the Decoder must learn to predict the end of a caption. This is necessary because we need to know when to stop decoding during inference.

`<start> a man holds a football <end>`

Since we pass the captions around as fixed size Tensors, we need to pad captions (which are naturally of varying length) to the same length. So we will also use `<pad>` tokens. We choose this fixed length to simply be the length of the longest caption in the corpus.

`<start> a man holds a football <end> <pad> <pad> <pad>....`

Furthermore, we create a `word_map` which is an index mapping for each word in the corpus, including the `<start>`,`<end>`, and `<pad>` tokens. PyTorch, like other libraries, needs words encoded as indices to look up embeddings for them or to identify their place in the predicted word scores.

`9876 1 5 120 1 5406 9877 9878 9878 9878....`

To summarize, the captions fed to the model must be an `Int` tensor of dimension `N, L` where `L` is the padded length.

##### Caption Lengths

Since the captions are padded, we need to keep track of the lengths of each caption. This is actual length + 2 (for the `<start>` and `<end>` pads).

Caption lengths are also important because of PyTorch's dynamic graphs. We only process a sequence upto its length and we don't need to waste any computation over the `<pad>`s.

Caption lengths fed to the model must be an `Int` tensor of dimension `N`.

### Data pipeline



### Encoder

### Soft Attention

### Decoder

# Training

### Loss Function

### Early stopping with BLEU

# Evaluation

### Beam search

### Some more examples

![](./img/plane.png)

![](./img/birds.png)

<br>
It samples the color at `white`, ignores the food at `plate`, and localizes the `salad`. Notice how it says `plate topped with` which are not words humans would use in this scenario, although it is technically correct. Must be all the pizza in the training data.

![](./img/salad.png)

![](./img/manbike.png)

![](./img/firehydrant.png)

<br>
It identifies the objects correctly (the `kitten`'s actually a cat), but it says `playing` instead of eating. You can see why it would think this - kittens do tend to play with things, and the model doesn't have the same real word experience to guess that bananas are food more often than playthings.
![](./img/catbanana.png)

<br> The ~~Turing~~ Tommy Test: you know AI's not really AI because it hasn't watched _The Room_ and doesn't recognize greatness when it sees it.
![](./img/tommy.png)

# FAQs

__You keep saying__ ___soft___ __attention. Is there, um, a__ ___hard___ __attention?__

__You mentioned I can use the same attention mechanism for sequence-to-sequence NLP tasks such as machine translation. How?__

__Why not use Beam Search during training?___

__What about not using Teacher Forcing during validation?__

__Can I use pretrained word embeddings (GloVe, CBOW, skipgram, etc.) instead of learning them from scratch?__

__How do I calculate Meteor or Ciderr scores?__
