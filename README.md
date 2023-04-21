# Image_Caption_Generator
Project Done as a part of requirements of Graduation of Udacity computer Vision Nanodegree. 
	Pre-processed the images in the MS COCO Dataset using PyTorch Transforms and converted the captions in the training set into sequence of integers using BOW vocabulary dictionary with a vocabulary threshold of 5. Defined and trained CNN encoder and LSTM Decoder on top of a time distributed embedding layer by using pre-trainedRESNET50 model as a feature extractor to encode an input image to convert into a fixed embed sized vector and then generate caption from the output embedding vector of the CNN encoder. Configurations of the data pre-processing and CNN encoder and LSTM decoder were inspire from the paper "https://arxiv.org/pdf/1411.4555.pdf". The inference is done on the 'test' portion of the MS COCO dataset.


