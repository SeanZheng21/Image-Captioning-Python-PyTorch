# Image Captioning Python PyTorch

## Machine learning Image caption generator using Python PyTorch - Text and Visual Data Drilling

![Caption 1](https://github.com/SeanZheng21/Image-Captioning-Python-PyTorch/blob/main/images/caption%201.png)

![Caption 2](https://github.com/SeanZheng21/Image-Captioning-Python-PyTorch/blob/main/images/caption%202.png)

Image captioning is very broadly used in many mainstream technologies like image indexing and virtual assistants. Visually impaired people and mobile voice assistant users can benefit from virtual assistants with image captioning ability. Image captioning models take advantage of powerful deep neural networks and generate rich text descriptions from images. 

In this study, we explore a hybrid architecture that consists of a CNN-based encoder and an RNN-based decoder. This architecture is able to leverage the intrinsic nature of multi-modalities for this task involving both visual elements and language. Additionally, we verify the feasibility of some popular attention modules, which improves the caption quality by assisting the decoder to focus on the most relevant regions. 

Furthermore, to fully exploit the image dataset with multiple descriptions, we propose to use a recently proposed contrastive learning strategy to train the network with sentence embedding during initialization in training. 

Our image captioning approach was tested out on the large-scale and high-quality Microsoft COCO image dataset. On this dataset, our proposed model is able to effectively generate image captions that properly describe the contents of the image. With our image captioning approach, this study contributes to the topic of image captioning by providing an experimentally proven and fully functional image captioning model. Based on our analysis of the experiment results, the future work of this study includes discovering the different impacts of various attention mechanisms, especially for the multi-head attention mechanism.

## Index Terms
* image understanding
* image captioning
* multi- modal prediction
* natural language processing

## Machine Learning Model and pipeline

A training diagram is shown in the following image, where images are fed into a CNN as a feature extractor and their output is fed into an RNN-based decoder to generate plausible sentences that describe the content of the input image. This type of architecture becomes the default one for the image captioning task. 

To further enhance the expressiveness of the RNN- based decoder, attention mechanism was proposed to selectively prune the feature maps given the current input word. Attention mechanism stemming from the study of human vision, allows to dynamically focus relevant regions given a feature map while removing redundant information that may hurt model optimization. 

Two types of attention were introduced: hard attention and soft attention. The hard attention is a stochastic discrete attention that can be approximated by reinforce algorithm, while the soft one is a deterministic function to be learned by scholastic gradient descent, resulting in an easier optimization.

![Image Pipeline](https://github.com/SeanZheng21/Image-Captioning-Python-PyTorch/blob/main/images/Image%20Pipeline.png)

## Caption Generation Model

Different from image classification tasks, language generation has one particular characteristic: one image may correspond to multiple ground truth target sentences. This one-to-many mapping can be difficult to learn and evaluate. Instead, recent work uses this multi-target nature to effectuate sentence embedding pretraining. 

Sentence embedding is a high-dimensional embedding that encodes the schematic meaning of the global sentences. Two sentences describing the same scene should have very similar embeddings, while if these sentences are taken from two different images, their embedding should be far away. Contrastive training with this technique was demonstrated to bring an impressive performance boost in neural machine translation and we aim to use this approach to boost the performance of the image captioning method, where multiple target sentences coexist for an image.

![Caption Model](https://github.com/SeanZheng21/Image-Captioning-Python-PyTorch/blob/main/images/Caption%20Model.png)

## Experiment

Dataset: Several datasets are available for the task of image captioning and we use the Coco dataset to verify the feasibility of the attention-based algorithm. The Coco dataset is a large-scale multi-propose dataset for object detection, segmentation and captioning, and has been widely used for tasks such as scene understanding. 

In a total of about 328k images, there are around 2.5 million labeled instances. The training set for our image captioning program contains 83 thousand images with a total size of 13 gigabytes. The validation set for our image captioning program contains 41 thousand images with a total size of 6 gigabytes. The testing set for our image captioning program contains 81 thousand images with a total size of 16 gigabytes. Every image has 5 human-generated captions.

We used a pre-trained resnet152 as our CNN backbone, serving to extract the feature representation of given images.

These are the visual inspection on the attention mechanism that allows us to analyze in detail how the model is generating sentences.

![Caption 1](https://github.com/SeanZheng21/Image-Captioning-Python-PyTorch/blob/main/images/caption%201.png)

![Caption 2](https://github.com/SeanZheng21/Image-Captioning-Python-PyTorch/blob/main/images/caption%202.png)


## BLUE Scores

In all settings, we get similar results in terms of various BLUE scores, but the pre-training-based one gives slightly better performance.

![Score](https://github.com/SeanZheng21/Image-Captioning-Python-PyTorch/blob/main/images/score.png)