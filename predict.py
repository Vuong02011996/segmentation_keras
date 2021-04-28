from keras_segmentation.models.unet import vgg_unet
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


model = vgg_unet(n_classes=19,  input_height=416, input_width=256)
# Display the model's architecture
model.summary()

model.load_weights("models/vgg_unet_1.99")

out = model.predict_segmentation(
    inp="dataset2/images_prepped_test/person_240_0.png",
    out_fname="out1.png"
)
# print(model.evaluate_segmentation(inp_images_dir="dataset2/images_prepped_test/", annotations_dir="dataset2/annotations_prepped_test_convert/"))