import glob
import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants



def main():

    model = "model/ieee_combined"  # load the model
    save_dir = "encoded_ieee_image/ieee_combined"  # save dir
    image_input_dir = "./doc_ieee_image"  # image input dir
    image_input_list = glob.glob(image_input_dir + '/*')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    sess = tf.InteractiveSession(graph=tf.Graph())
    model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], model)

    input_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['secret'].name
    input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
    input_secret = tf.get_default_graph().get_tensor_by_name(input_secret_name)
    input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

    output_encoded_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['encoded'].name
    output_encoded = tf.get_default_graph().get_tensor_by_name(output_encoded_name)

    size = (400, 400)
    secret_binary = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                     1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                     0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1]
    secret = [int(x) for x in secret_binary]

    for image_idx in range(100):

        image = cv2.imread(filename=image_input_list[image_idx], flags=cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(cv2.resize(image, size), dtype=np.float32)
        image /= 255.

        feed_dict = {input_secret: [secret], input_image: [image]}
        hidden_image = sess.run(output_encoded, feed_dict=feed_dict)

        encoded_image = (hidden_image[0] * 255).astype(np.uint8)
        encoded_image = cv2.cvtColor(encoded_image, cv2.COLOR_RGB2BGR)
        save_name = image_input_list[image_idx].split('\\')[-1].split('.')[0]
        quality_factor = 100.0
        cv2.imwrite(save_dir + '/' + str(save_name) + '_hidden.jpeg', encoded_image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])

    print("===================================")
    print("Saving over！")


if __name__ == "__main__":
    main()
