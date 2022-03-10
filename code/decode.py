import glob
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from albumentations import ImageCompression, GaussianBlur, GaussNoise, Resize
import utils


def main():
    model = "model/ieee_combined"  # load the model
    image_input_dir = "./doc_ieee_image"  # ./doc_cnki_image
    image_encoded_dir = "encoded_ieee_image/ieee_combined"

    sess = tf.InteractiveSession(graph=tf.Graph())
    model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], model)

    input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
    input_encoded_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

    output_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs[
        'decoded'].name
    output_secret = tf.get_default_graph().get_tensor_by_name(output_secret_name)

    width = height = 400
    channel = 3
    count_total = 0
    secret_binary = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                     1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                     0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1]
    size = (width, height)

    image_input_list = glob.glob(image_input_dir + '/*')

    image_encoded_list = glob.glob(image_encoded_dir + '/*')
    for image_idx in range(len(image_encoded_list)):

        # encoded_image
        encode_image = cv2.imread(filename=image_encoded_list[image_idx], flags=cv2.IMREAD_UNCHANGED)
        encode_image = cv2.cvtColor(encode_image, cv2.COLOR_BGR2RGB)
        arr_encode_image = np.array(cv2.resize(encode_image, size), dtype=np.float32)
        encoded_image = arr_encode_image / 255.

        # image_input
        origin_image = cv2.imread(filename=image_input_list[image_idx], flags=cv2.IMREAD_UNCHANGED)
        origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
        arr_input_image = np.array(cv2.resize(origin_image, size), dtype=np.float32)
        input_image = arr_input_image / 255.

        # Dropout
        dropout_probability = 0.1
        mask2d = tf.random.uniform(encoded_image.shape[0:2], minval=0.0, maxval=1.0, dtype=tf.float32)
        mask3d = tf.broadcast_to(tf.expand_dims(mask2d, -1), shape=encoded_image.shape[0:])
        dropout_mask = tf.math.ceil(tf.clip_by_value(mask3d - (1 - dropout_probability), 0, 1))
        dropout_image = encoded_image * (1 - dropout_mask) + input_image * dropout_mask
        hidden_image2 = dropout_image.eval()

        # Cropout
        cropout_proportion = 0.9
        crop_width = tf.cast(tf.math.sqrt((width * height * (1 - cropout_proportion))), tf.int32)
        x_start = tf.random.uniform((), minval=0, maxval=width - crop_width, dtype=tf.int32)
        y_start = tf.random.uniform((), minval=0, maxval=height - crop_width, dtype=tf.int32)
        cropout_mask = tf.ones((crop_width, crop_width, channel))
        cropout_mask = tf.image.pad_to_bounding_box(cropout_mask, y_start, x_start, width, height)
        cropout_image = encoded_image * (1 - cropout_mask) + input_image * cropout_mask
        hidden_image2 = cropout_image.eval()

        # Gaussian Blur
        encoded_image = tf.reshape(encoded_image, (1, height, width, channel))
        filter_blur = utils.random_blur_kernel(probs=0.5, blur_kernel=7, sigrange_gauss=[1., 3.])
        conv2d_image = tf.nn.conv2d(encoded_image, filter_blur, [1, 1, 1, 1], padding='SAME')
        hidden_image2 = conv2d_image[0].eval()

        # Gaussian Blur 3->1
        hidden_image2 = GaussianBlur(blur_limit=7, always_apply=True, p=1.0)(image=encoded_image)['image']
        hidden_image2 = cv2.GaussianBlur(encoded_image, (7, 7), 0)

        # Gaussian Noise 0.02->0.05
        rnd_noise = 0.05
        encoded_noise = tf.random_normal(shape=tf.shape(encoded_image), mean=0.0, stddev=rnd_noise, dtype=tf.float32)
        noise_image = encoded_image + encoded_noise
        noised_image = tf.clip_by_value(noise_image, 0, 1)
        hidden_image2 = noised_image.eval()

        # Resize 50->10
        resize_proportion = 0.1
        image_size = int(resize_proportion * 400)
        resize_image = Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR, p=1.0)(image=encoded_image)['image']
        hidden_image2 = Resize(400, 400, interpolation=cv2.INTER_LINEAR, p=1.0)(image=resize_image)['image']

        # Jpeg Compression 50->30
        quality_upper = 10
        quality_lower = quality_upper - 1
        hidden_image2 = ImageCompression(quality_lower=quality_lower, quality_upper=quality_upper, p=1.0)(image=encoded_image)['image']

        hidden_image = (hidden_image2 * 255.).astype(np.uint8)
        encoded_image_2 = np.array(cv2.resize(hidden_image, size), dtype=np.float32) / 255.

        feed_dict = {input_encoded_image: [encoded_image_2]}
        secret_extract = sess.run([output_secret], feed_dict=feed_dict)[0][0]

        for i in range(len(secret_extract)):
            if secret_binary[i] == int(secret_extract[i]):
                count_total += 1

    accuracy_total = count_total / len(image_encoded_list)
    print("===================================")
    print("Total document image accuracy is:{0}%".format(accuracy_total))


if __name__ == "__main__":
    main()
