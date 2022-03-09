import numpy as np
import tensorflow as tf
import utils
import cv2
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *


# Encoder architecture
class wtrmarkEncoder(Layer):
    def __init__(self):
        super(wtrmarkEncoder, self).__init__()

        self.secret_dense = Dense(7500, activation='relu', kernel_initializer='he_normal')
        self.conv1 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv2 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv3 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv5 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv6 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv7 = Conv2D(3, (3, 3), activation=None, padding='same', kernel_initializer='he_normal')

    def call(self, inputs):
        secret, image = inputs

        secret = self.secret_dense(secret)
        secret = Reshape((50, 50, 3))(secret)
        secret_enlarged = UpSampling2D(size=(8, 8))(secret)

        inputs = concatenate([secret_enlarged, image], axis=-1)
        conv1 = self.conv1(inputs)

        merge1 = concatenate([secret_enlarged,conv1], axis=3)
        conv2 = self.conv2(merge1)

        merge2 = concatenate([inputs,conv2], axis=3)
        conv3 = self.conv3(merge2)

        merge3 = concatenate([secret_enlarged, conv3], axis=3)
        conv4 = self.conv4(merge3)

        merge4 = concatenate([secret_enlarged, conv4], axis=3)
        conv5 = self.conv5(merge4)

        merge5 = concatenate([inputs, conv5], axis=3)
        conv6 = self.conv6(merge5)

        merge6 = concatenate([secret_enlarged, conv6], axis=3)
        mask = self.conv7(merge6)

        return mask


# Decoder architecture
class wtrmarkDecoder(Layer):
    def __init__(self, secret_size, height, width):
        super(wtrmarkDecoder, self).__init__()

        self.decoder = Sequential([
            Conv2D(16, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal'),
            Conv2D(16, (3, 3), strides=2, activation='relu', padding='same',kernel_initializer='he_normal'),
            Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal'),
            Conv2D(32, (3, 3), strides=2, activation='relu', padding='same',kernel_initializer='he_normal'),
            Conv2D(32, (3, 3), strides=2, activation='relu', padding='same',kernel_initializer='he_normal'),
            Conv2D(64, (3, 3), strides=2, activation='relu', padding='same',kernel_initializer='he_normal'),
            Conv2D(128, (3, 3), strides=2, activation='relu', padding='same',kernel_initializer='he_normal'),
            Flatten(),
            Dense(secret_size)
        ])

    def call(self, image):
        decoded_secret = self.decoder(image)
        return decoded_secret


# noise_layer
def noise_layer(encoded_image, image_input, global_step):

    height = width = 400
    channel = 3

    ramp_fn = lambda ramp: tf.minimum(tf.to_float(global_step) / ramp, 1.)  # (?/1000,1.0) = (0,1.0) ?:[0,100000]

    # Dropout 0.1
    dropout_probability = tf.random.uniform([], 0., 0.1)
    mask2d = tf.random.uniform(encoded_image.shape[1:3], minval=0.0, maxval=1.0, dtype=tf.float32)
    mask3d = tf.broadcast_to(tf.expand_dims(mask2d, -1), shape=encoded_image.shape[1:])
    dropout_mask = tf.math.ceil(tf.clip_by_value(mask3d - dropout_probability, 0, 1))
    encoded_image = encoded_image * dropout_mask + image_input * (1 - dropout_mask)

    # Cropout 0.1
    cropout_proportion = tf.random.uniform([], 0., 0.1)
    crop_width = tf.cast(tf.math.sqrt((width * height * (1.0-cropout_proportion))), tf.int32)
    x_start = tf.random.uniform((), minval=0, maxval=width - crop_width, dtype=tf.int32)
    y_start = tf.random.uniform((), minval=0, maxval=height - crop_width, dtype=tf.int32)
    cropout_mask = tf.ones((crop_width, crop_width, channel))
    cropout_mask = tf.image.pad_to_bounding_box(cropout_mask, y_start, x_start, width, height)
    encoded_image = encoded_image * (1 - cropout_mask) + image_input * cropout_mask

    # Gaussian Blur
    filter_blur = utils.random_blur_kernel(probs=0.5, blur_kernel=3, sigrange_gauss=[1., 3.])
    encoded_image = tf.nn.conv2d(encoded_image, filter_blur, [1, 1, 1, 1], padding='SAME')

    # Gaussian Noise 0.02
    rnd_noise = 0.02
    rnd_noise_ramp = 1000
    rnd_noise = tf.random.uniform([]) * ramp_fn(rnd_noise_ramp) * rnd_noise
    noise = tf.random_normal(shape=tf.shape(encoded_image), mean=0.0, stddev=rnd_noise, dtype=tf.float32)
    encoded_image = encoded_image + noise
    encoded_image = tf.clip_by_value(encoded_image, 0, 1)

    # Resize
    rnd_resize = tf.random.uniform([], 0.5, 1.0)
    height_resize = rnd_resize * height
    width_resize = rnd_resize * width
    resize_image = tf.image.resize_images(encoded_image, (height_resize, width_resize), method=0)
    encoded_image = tf.image.resize_images(resize_image, (height, width), method=0)

    # Jpeg Compression
    jpeg_quality = 50
    jpeg_quality_ramp = 1000
    jpeg_quality = 100. - tf.random.uniform([]) * ramp_fn(jpeg_quality_ramp) * (100. - jpeg_quality)
    jpeg_factor = tf.cond(tf.less(jpeg_quality, 50), lambda: 5000. / jpeg_quality, lambda: 200. - jpeg_quality * 2) / 100. + .0001
    encoded_image = utils.jpeg_compress_decompress(encoded_image, rounding=utils.round_only_at_0, factor=jpeg_factor, downsample_c=True)

    return encoded_image


# build model
def build_model(encoder,
                decoder,
                secret_input,
                image_input,
                factor,
                loss_scales,
                yuv_scales,
                rgb_scales,
                global_step):

    # encoder
    mask = encoder((secret_input, image_input))
    """
      Embedding strength adjustment strategy is meant to change the factor to get high visual quality of the document image.
      you can set the factor value before the mode training, or the default value is 1.0, 
      the more detail you can read the paper from the link: 
      https://www.semanticscholar.org/paper/A-Robust-Document-Image-Watermarking-Scheme-using-Ge-Xia/8bc9b8d124cbaf5ad516c0521aecad5f3d45e604
    """
    encoded_image = image_input + factor * mask

    # noise layer
    noised_image = noise_layer(encoded_image, image_input, global_step)

    # decoder
    decoded_secret = decoder(noised_image)  # encoded_image

    encoded_image_yuv = tf.image.rgb_to_yuv(encoded_image)
    image_input_yuv = tf.image.rgb_to_yuv(image_input)
    im_yuv_diff = encoded_image_yuv - image_input_yuv
    yuv_loss_op = tf.reduce_mean(tf.square(im_yuv_diff), axis=[0, 1, 2])
    # image_loss
    image_loss_op = tf.tensordot(yuv_loss_op, yuv_scales, axes=1)

    # secret_loss
    secret_loss_op = tf.losses.sigmoid_cross_entropy(secret_input, decoded_secret)

    im_rgb_diff = tf.abs(encoded_image - image_input)
    im_text_diff = im_rgb_diff * tf.convert_to_tensor(1.0 - image_input, dtype=tf.float32)
    diff_loss_op = tf.reduce_mean(im_text_diff, axis=[0, 1, 2])
    # text_loss
    text_loss_op = tf.tensordot(diff_loss_op, rgb_scales, axes=1)

    loss_op = loss_scales[0] * image_loss_op  + loss_scales[1] * secret_loss_op + loss_scales[2] * text_loss_op

    return loss_op, image_loss_op, secret_loss_op, text_loss_op


# prepare deployment hiding graph
def prepare_deployment_hiding_graph(encoder, secret_input, image_input):

    residual = encoder((secret_input, image_input))
    encoded_image = image_input + residual
    encoded_image = tf.clip_by_value(encoded_image, 0, 1)

    return encoded_image, residual


# prepare deployment reveal graph
def prepare_deployment_reveal_graph(decoder, image_input):

    decoded_secret = decoder(image_input)
    secret = tf.round(tf.sigmoid(decoded_secret))

    return secret
