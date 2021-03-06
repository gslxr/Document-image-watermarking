import glob
import os
import numpy as np
import tensorflow as tf
import utils
import models
from os.path import join

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# CNKI/IEEE
TRAIN_PATH = './imgs/ieee/'
MODELS_PATH = './models/'
SAVED_MODELS = './saved_models/ieee'


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', type=str, default='mTrain')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--secret_size', type=int, default=100)
    parser.add_argument('--num_steps', type=int, default=140000)
    parser.add_argument('--no_im_loss_steps', type=int, default=1500)
    parser.add_argument('--rnd_trans', type=float, default=0.1)
    parser.add_argument('--rnd_trans_ramp', type=int, default=10000)
    parser.add_argument('--secret_loss_scale', type=float, default=1.0)
    parser.add_argument('--secret_loss_ramp', type=int, default=1)
    parser.add_argument('--l2_loss_scale', type=float, default=1.5)
    parser.add_argument('--l2_loss_ramp', type=int, default=20000)
    parser.add_argument('--text_loss_scale', type=float, default=1)
    parser.add_argument('--text_loss_ramp', type=int, default=20000)
    parser.add_argument('--factor', type=float, default=1.0)
    parser.add_argument('--y_scale', type=float, default=1.0)
    parser.add_argument('--u_scale', type=float, default=100.0)
    parser.add_argument('--v_scale', type=float, default=100.0)
    parser.add_argument('--r_scale', type=float, default=0.3)
    parser.add_argument('--g_scale', type=float, default=0.6)
    parser.add_argument('--b_scale', type=float, default=0.1)
    args = parser.parse_args()

    EXP_NAME = args.exp_name
    files_list = glob.glob(join(TRAIN_PATH, "**/*"))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    height = 400
    width = 400

    secret_pl = tf.placeholder(shape=[None, args.secret_size], dtype=tf.float32, name="input_prep")
    image_pl = tf.placeholder(shape=[None, height, width, 3], dtype=tf.float32, name="input_hide")
    global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
    factor_pl = tf.placeholder(shape=[1], dtype=tf.float32, name="factor")
    loss_scales_pl = tf.placeholder(shape=[3], dtype=tf.float32, name="input_loss_scales")
    yuv_scales_pl = tf.placeholder(shape=[3], dtype=tf.float32, name="input_yuv_scales")
    rgb_scales_pl = tf.placeholder(shape=[3], dtype=tf.float32, name="input_rgb_scales")

    encoder = models.wtrmarkEncoder()
    decoder = models.wtrmarkDecoder(secret_size=args.secret_size, height=height, width=width)

    loss_op, image_loss_op, secret_loss_op, text_loss_op = models.build_model(
        encoder=encoder,
        decoder=decoder,
        secret_input=secret_pl,
        image_input=image_pl,
        factor=factor_pl,
        loss_scales=loss_scales_pl,
        yuv_scales=yuv_scales_pl,
        rgb_scales=rgb_scales_pl,
        global_step=global_step_tensor)

    tvars = tf.trainable_variables()

    g_vars = [var for var in tvars if 'wtrmark' in var.name]

    train_loss_op = tf.train.AdamOptimizer(args.lr).minimize(loss_op, var_list=g_vars, global_step=global_step_tensor)
    train_secret_op = tf.train.AdamOptimizer(args.lr).minimize(secret_loss_op, var_list=g_vars, global_step=global_step_tensor)

    deploy_hide_image_op, residual_op = models.prepare_deployment_hiding_graph(encoder, secret_pl, image_pl)
    deploy_decoder_op = models.prepare_deployment_reveal_graph(decoder, image_pl)

    total_steps = len(files_list) // args.batch_size + 1
    global_step = 0

    saver = tf.train.Saver(tvars, max_to_keep=100)
    sess.run(tf.global_variables_initializer())

    if args.pretrained is not None:
        saver.restore(sess, args.pretrained)

    while global_step < args.num_steps:
        for _ in range(min(total_steps, args.num_steps - global_step)):
            no_im_loss = global_step < args.no_im_loss_steps
            images, secrets = utils.get_img_batch(files_list=files_list,
                                                  secret_size=args.secret_size,
                                                  batch_size=args.batch_size,
                                                  size=(height, width))

            l2_loss_scale = min(args.l2_loss_scale * global_step / args.l2_loss_ramp, args.l2_loss_scale)  # (2*100000/15000,2) = (0,2)
            secret_loss_scale = min(args.secret_loss_scale * global_step / args.secret_loss_ramp, args.secret_loss_scale)  # (2*150000/15000,2) = (0,2)
            text_loss_scale = min(args.text_loss_scale * global_step / args.text_loss_ramp, args.text_loss_scale)  # (1.5*100000/15000,1.5) = (0,1.5)

            feed_dict = {secret_pl: secrets,
                         image_pl: images,
                         factor_pl: args.factor,
                         loss_scales_pl: [l2_loss_scale, secret_loss_scale, text_loss_scale],
                         yuv_scales_pl: [args.y_scale, args.u_scale, args.v_scale],
                         rgb_scales_pl: [args.r_scale, args.g_scale, args.b_scale]}

            if no_im_loss:
                _, total_loss, image_loss, secret_loss, text_loss, global_step = sess.run([train_secret_op, loss_op, image_loss_op, secret_loss_op, text_loss_op, global_step_tensor], feed_dict)
                if global_step % 100 == 0:
                    print('global_step:', global_step, 'total_loss:', total_loss, 'image_loss:', image_loss, 'secret_loss:', secret_loss, 'text_loss:', text_loss)
            else:
                _, total_loss, image_loss, secret_loss, text_loss, global_step = sess.run([train_loss_op, loss_op, image_loss_op, secret_loss_op, text_loss_op, global_step_tensor], feed_dict)
                if global_step % 100 == 0:
                    print('global_step:', global_step, 'total_loss:', total_loss, 'image_loss:', image_loss, 'secret_loss:', secret_loss, 'text_loss:', text_loss)

            if global_step % 50000 == 0:
                saver.save(sess, os.path.join(MODELS_PATH, EXP_NAME), global_step=global_step)

    constant_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph.as_graph_def(),
        [deploy_hide_image_op.name[:-2], residual_op.name[:-2], deploy_decoder_op.name[:-2]])
    with tf.Session(graph=tf.Graph()) as sess:
        tf.import_graph_def(constant_graph_def, name='')
        tf.saved_model.simple_save(sess,
                                   SAVED_MODELS + '/' + EXP_NAME,
                                   inputs={'secret': secret_pl, 'image': image_pl},
                                   outputs={'encoded': deploy_hide_image_op,
                                            'residual': residual_op, 'decoded': deploy_decoder_op})


if __name__ == "__main__":
    main()
