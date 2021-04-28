from keras_segmentation.pretrained import pspnet_50_ADE_20K , pspnet_101_cityscapes, pspnet_101_voc12
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

model = pspnet_50_ADE_20K() # load the pretrained model trained on ADE20k dataset
#
# model = pspnet_101_cityscapes() # load the pretrained model trained on Cityscapes dataset

# model = pspnet_101_voc12() # load the pretrained model trained on Pascal VOC 2012 dataset

# load any of the 3 pretrained models

out = model.predict_segmentation(
    inp="data/ADE_val_00001054.jpg",
    out_fname="out.png"
)