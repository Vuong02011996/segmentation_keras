from keras_segmentation.models.unet import vgg_unet
import matplotlib.pyplot as plt

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

model = vgg_unet(n_classes=19,  input_height=416, input_width=256)
# Display the model's architecture
model.summary()

model.train(
    train_images =  "dataset3/train/",
    train_annotations = "dataset3/label_convert/",
    checkpoints_path = "models/vgg_unet_1/",
    epochs=100,
)

out = model.predict_segmentation(
    inp="dataset2/images_prepped_test/person_240_0.png",
    out_fname="out.png"
)
print(out)
plt.imshow(out)

# # evaluating the model
# print(model.evaluate_segmentation( inp_images_dir="dataset2/images_prepped_test/",
#                                    annotations_dir="dataset2/annotations_prepped_test_convert/" ))

# python -m keras_segmentation predict \
#  --checkpoints_path="/home/vuong/Desktop/Project/MyGitHub/segmentation_keras/models/vgg_unet_1/19" \
#  --input_path="/home/vuong/Desktop/Project/MyGitHub/segmentation_keras/dataset2/images_prepped_test/" \
#  --output_path="/home/vuong/Desktop/Project/MyGitHub/segmentation_keras/output"
