import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras as keras

from PIL import Image
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv1D, Conv2D, Conv2DTranspose, Conv3D, Conv3DTranspose, Dense, Flatten, Input, LayerNormalization, LeakyReLU, Reshape)

mse_loss = keras.losses.MeanSquaredError()
bce_loss_logits = keras.losses.BinaryCrossentropy(from_logits=True)

def minus_1_plus_1_to_uint8_arr(x):
    return ((x + 1) * 127.5).astype(np.uint8)

def uint8_to_minus_1_plus_1_arr(x):
    return (x.astype(np.float32) / 127.5) - 1.0

def save_video_as_gif(video, filename='video'):
    video = [Image.fromarray(frame) for frame in video]
    video[0].save(f'{filename}.gif', format='GIF', append_images=video[1:], save_all=True, duration=100, loop=0)

def save_minus_1_plus_1_video_as_gif(video, *args, **kwargs):
    video = minus_1_plus_1_to_uint8_arr(video)
    save_video_as_gif(video, *args, **kwargs)

def resize_image(image, width, height, resampler=Image.LANCZOS):
    return np.array(Image.fromarray(image).resize((width, height), resample=resampler))

def resize_minus_1_plus_1(image, *args, **kwargs):
    image = minus_1_plus_1_to_uint8_arr(image)
    image = resize_image(image, *args, **kwargs)
    return uint8_to_minus_1_plus_1_arr(image)

def load_images_from_dir(images_dir):
    categories = sorted(os.listdir(images_dir))
    category_to_label = {c: i for i, c in enumerate(categories)}

    images = []
    labels = []

    for category in categories:
        category_label = category_to_label[category]
        category_path = os.path.join(images_dir, category)

        print('loading images for category:', category, ', label:', category_label, ', path:', category_path)

        image_filenames = sorted(os.listdir(category_path))

        for image_filename in image_filenames:
            img_path = os.path.join(category_path, image_filename)
            img = uint8_to_minus_1_plus_1_arr(plt.imread(img_path)[:, :, :3])
            images.append(img)
            labels.append(category_label)

    return images, labels, categories, category_to_label

def prepare_video_dataset(video_images, video_labels, fixed_size=64, min_vid_len=16):
    x_imgs = []
    y_imgs = []

    x_vids = []
    y_vids = []

    for video_image, video_label in zip(video_images, video_labels):
        H, W, _ = video_image.shape

        if W != fixed_size and H != fixed_size:
            print(f'video image must have width or height = {fixed_size}. actual shape: {video_image.shape}, label: {video_label}')
            continue

        horizontal = W >= H

        video_length = W // H if horizontal else H // W

        if video_length < min_vid_len:
            print(f'video must have minimum {min_vid_len} frames. actual length: {video_length}, video image shape: {video_image.shape}, label: {video_label}')
            continue

        video = np.split(video_image, video_length, axis=1 if horizontal else 0)

        x_imgs.extend(video)
        y_imgs.extend([video_label]*video_length)

        x_vids.append(np.array(video))
        y_vids.append(video_label)

    return np.array(x_imgs), np.array(y_imgs, dtype=np.uint8), np.array(x_vids), np.array(y_vids, dtype=np.uint8)

def load_actions_dataset(drive_dir=os.path.join('/', 'content', 'drive')):
    DATASET_DIR = os.path.join(drive_dir, 'My Drive', 'Colab Notebooks', 'thesis', 'gan', 'video', 'mocogan', 'actions')

    if not os.path.isdir(DATASET_DIR):
        return print('dataset directory not found. DATASET_DIR:', DATASET_DIR)

    video_images, video_labels, categories, category_to_label = load_images_from_dir(DATASET_DIR)

    return (*prepare_video_dataset(video_images, video_labels), categories, category_to_label)

def create_up_sampler(input_shape, output_shape, activation=None, num_filters=512, min_filters=16, regular_sizes=True, use_batchnorm=True):
    """
    This function creates a up sampler that takes in 3d tensors of shape input_shape and produces 3d tensors of shape output_shape. 
    3d tensor shape should be of the form (height, width, channels). example: (64, 64, 3)

    activation is the activation to be used in the final layer.

    num_filters is the no. of filters to start with.

    min_filters is the minimum no. of filters any Conv2DTranspose will have.

    if regular_sizes is True then width, height, num_filters and min_filters should be a power of 2

    if use_batchnorm is True then batch normalization is used, otherwise layer normalization is used.
    if using gradient penalty batch normalization cannot be used.
    """

    iH, iW, iC = input_shape
    oH, oW, oC = output_shape

    assert int(np.log2(oW // iW)) == int(np.log2(oH // iH)) # no. of up sample steps should be same for both width and height

    for i_size, o_size in zip([iH, iW, min_filters], [oH, oW, num_filters]):
        assert i_size <= o_size

    if regular_sizes:
        supported_sizes = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}
        for size in [iH, iW, min_filters, oH, oW, num_filters]:
            assert size in supported_sizes

    model = Sequential([Input(shape=input_shape)])

    while iH < oH:
        model.add(Conv2DTranspose(num_filters, 4, 2, 'same', use_bias=False))
        if use_batchnorm:
            model.add(BatchNormalization())
        else:
            model.add(LayerNormalization(axis=[-3, -2, -1]))
        model.add(Activation('relu'))
        iH *= 2
        if num_filters > min_filters:
            num_filters //= 2

    model.add(Conv2DTranspose(oC, 3, 1, 'same', use_bias=True))

    if activation is not None:
        model.add(Activation(activation))

    assert model.output_shape == (None, *output_shape)

    return model

def create_down_sampler(input_shape, output_shape, activation=None, num_filters=128, max_filters=512, regular_sizes=True, use_batchnorm=True):
    """
    This function creates a down sampler that takes in 3d tensors of shape input_shape and produces 3d tensors of shape output_shape. 
    3d tensor shape should be of the form (height, width, channels). example: (64, 64, 3)

    activation is the activation to be used in the final layer.

    num_filters is the no. of filters to start with.

    max_filters is the maximum no. of filters any Conv2D will have.

    if regular_sizes is True then width, height, num_filters and max_filters should be a power of 2

    if use_batchnorm is True then batch normalization is used, otherwise layer normalization is used.
    if using gradient penalty batch normalization cannot be used.
    """

    iH, iW, iC = input_shape
    oH, oW, oC = output_shape

    assert int(np.log2(iW // oW)) == int(np.log2(iH // oH)) # no. of down sample steps should be same for both width and height

    for i_size, o_size in zip([iH, iW, max_filters], [oH, oW, num_filters]):
        assert i_size >= o_size

    if regular_sizes:
        supported_sizes = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}
        for size in [iH, iW, max_filters, oH, oW, num_filters]:
            assert size in supported_sizes

    model = Sequential([Input(shape=input_shape)])

    model.add(Conv2D(num_filters, 4, 1, 'same'))
    model.add(LeakyReLU())
    if num_filters < max_filters:
        num_filters *= 2

    while iH > oH:
        model.add(Conv2D(num_filters, 4, 2, 'same'))
        if use_batchnorm:
            model.add(BatchNormalization())
        else:
            model.add(LayerNormalization(axis=[-3, -2, -1]))
        model.add(LeakyReLU())
        iH //= 2
        if num_filters < max_filters:
            num_filters *= 2

    model.add(Conv2D(oC, 4, 1, 'same'))

    if activation is not None:
        model.add(Activation(activation))

    assert model.output_shape == (None, *output_shape)

    return model

"""# TESTS"""

def test_create_up_sampler():
    tmod = create_up_sampler((1, 1, 100), (64, 64, 3), activation='tanh')
    print(type(tmod), tmod.input_shape, tmod.output_shape)
#     tmod.summary()

    ti = tf.random.normal((16, 1, 1, 100))
    print(ti)
    to = tmod(ti, training=False)
    print(to)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        to = sess.run(to)

    plt.subplot(1, 2, 1)
    plt.imshow(utils.minus_1_plus_1_to_uint8_arr(to[0]))

    tmod = create_up_sampler((7, 7, 100), (28, 28, 1), regular_sizes=False)
    print(type(tmod), tmod.input_shape, tmod.output_shape)
#     tmod.summary()

    ti = tf.random.normal((16, 7, 7, 100))
    print(ti.shape, ti.dtype)
    to = tmod(ti, training=False)
    print(to.shape, to.dtype)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        to = sess.run(to)

    plt.subplot(1, 2, 2)
    plt.imshow(utils.minus_1_plus_1_to_uint8_arr(to[0].squeeze()), cmap='gray')

def test_create_down_sampler():
    tmod = create_down_sampler((64, 64, 3), (1, 1, 100), num_filters=16, activation='tanh')
    print(type(tmod), tmod.input_shape, tmod.output_shape)
#     tmod.summary()

    ti = tf.random.normal((16, 64, 64, 3))
    print(type(ti), ti)
    to = tmod(ti, training=False)
    print(type(to), to)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(to)

    print(result.shape, result.dtype)
    print(result)

def test_combined():
    tgenerator = create_up_sampler((1, 1, 100), (64, 64, 3), activation='tanh', use_batchnorm=False)
    print(tgenerator.input_shape, tgenerator.output_shape)
#     tgenerator.summary()

    treconstructor = create_down_sampler((64, 64, 3), (1, 1, 100), num_filters=16)
    print(treconstructor.input_shape, treconstructor.output_shape)
#     treconstructor.summary()

    tdiscriminator = create_down_sampler((64, 64, 3), (1, 1, 1), num_filters=16, use_batchnorm=False)
    print(tdiscriminator.input_shape, tdiscriminator.output_shape)
#     tdiscriminator.summary()

    ti = tf.random.normal((4, 1, 1, 100))
    tm = tgenerator(ti, training=False)
    tj = treconstructor(tm, training=False)
    to = tdiscriminator(tm, training=False)

    print(ti.shape, tm.shape, tj.shape, to.shape)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        [ti, tm, tj, to] = sess.run([ti, tm, tj, to])

    for arr in [ti, tm, tj, to]:
        print(type(arr), arr.shape)

    plt.imshow(minus_1_plus_1_to_uint8_arr(tm[0]))
    print(tm[0, 0, :10, 0])
    print(np.all(np.isclose(ti, tj)))

def run_tests():
    from google.colab import drive
    drive.mount('/content/drive')

    x_imgs, y_imgs, x_vids, y_vids, categories, category_to_label = load_actions_dataset()

    print(categories, category_to_label)

    for arr in [x_imgs, y_imgs, x_vids, y_vids]:
        print(arr.shape, arr.dtype)

    print(x_vids[0].shape, x_vids[0].dtype)

    plt.axis('off')
    plt.imshow(minus_1_plus_1_to_uint8_arr(x_vids[0][0]))
    plt.title(f'{x_vids[0][0].shape}')
    plt.pause(0.05)

    save_minus_1_plus_1_video_as_gif(x_vids[0])

    resized = resize_minus_1_plus_1(x_imgs[0], 100, 100)

    plt.axis('off')
    plt.imshow(minus_1_plus_1_to_uint8_arr(resized))
    plt.title(f'original shape: {x_imgs[0].shape}, resized shape: {resized.shape}')
    plt.pause(0.05)

    test_combined()

if __name__ == "__main__":
    run_tests()
