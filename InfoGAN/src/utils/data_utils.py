from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import os
import h5py

import matplotlib.pylab as plt

from keras.preprocessing.image import ImageDataGenerator


def normalization(X):
    return X / 127.5 - 1


def inverse_normalization(X):
    return np.round((X + 1.) / 2. * 255.).astype('uint8')


def load_mnist(image_data_format):

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if image_data_format == "channels_first":
        X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
        X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    else:
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")

    X_train = normalization(X_train)
    X_test = normalization(X_test)

    nb_classes = len(np.unique(np.hstack((y_train, y_test))))

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    return X_train, Y_train, X_test, Y_test


def load_celebA(img_dim, image_data_format):

    with h5py.File("../../data/processed/CelebA_%s_data.h5" % img_dim, "r") as hf:

        X_real_train = hf["data"][:].astype(np.float32)
        X_real_train = normalization(X_real_train)

        if image_data_format == "channels_last":
            X_real_train = X_real_train.transpose(0, 2, 3, 1)

        return X_real_train


def gen_batch(X, batch_size):

    while True:
        idx = np.random.choice(X.shape[0], batch_size, replace=False)
        yield X[idx]


def data_generator_from_dir(data_dir, target_size, batch_size):

    # data_gen args
    print("Loading data from", data_dir)

    # Check if number of files in data_dir is a multiple of batch_size
    number_of_images = sum([len(files) for r, d, files in os.walk(data_dir)])
    if number_of_images % batch_size != 0:
        raise ValueError("ERROR: # of images in " + str(data_dir) + " found by keras.ImageDataGenerator is not a multiple of the batch_size ( " + str(batch_size) + " )!\nFound " + str(number_of_images) + " images. Add " + str(batch_size - number_of_images % batch_size) + " more image(s), or delete " + str(number_of_images % batch_size) + " image(s).")

    # datagens
    data_generator_args = dict(preprocessing_function=normalization)
    image_datagen = ImageDataGenerator(**data_generator_args)

    # Image generators
    image_data_generator = image_datagen.flow_from_directory(data_dir, target_size=target_size, batch_size=batch_size, class_mode=None, seed=29)

    if len(image_data_generator) == 0:
        raise ValueError("ERROR: # of images found by keras.ImageDataGenerator is 0!\nPlease save the images in the data_dir into at least one modre directory, preferably into classes. Given data_dir:", data_dir)

    return image_data_generator


def sample_noise(noise_scale, batch_size, noise_dim):

    return np.random.normal(scale=noise_scale, size=(batch_size, noise_dim[0]))


def sample_cat(batch_size, cat_dim):

    y = np.zeros((batch_size, cat_dim[0]), dtype="float32")
    random_y = np.random.randint(0, cat_dim[0], size=batch_size)
    y[np.arange(batch_size), random_y] = 1

    return y


def get_disc_batch(X_real_batch, generator_model, batch_counter, batch_size, cat_dim, cont_dim, noise_dim,
                   noise_scale=0.5, label_smoothing=False, label_flipping=0):

    # Create X_disc: alternatively only generated or real images
    if batch_counter % 2 == 0:
        # Pass noise to the generator
        y_cat = sample_cat(batch_size, cat_dim)
        y_cont = sample_noise(noise_scale, batch_size, cont_dim)
        noise_input = sample_noise(noise_scale, batch_size, noise_dim)
        # Produce an output
        X_disc = generator_model.predict([y_cat, y_cont, noise_input],batch_size=batch_size)
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
        y_disc[:, 0] = 1

        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    else:
        X_disc = X_real_batch
        y_cat = sample_cat(batch_size, cat_dim)
        y_cont = sample_noise(noise_scale, batch_size, cont_dim)
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
        if label_smoothing:
            y_disc[:, 1] = np.random.uniform(low=0.9, high=1, size=y_disc.shape[0])
        else:
            y_disc[:, 1] = 1

        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    # Repeat y_cont to accomodate for keras" loss function conventions
    y_cont = np.expand_dims(y_cont, 1)
    y_cont = np.repeat(y_cont, 2, axis=1)

    return X_disc, y_disc, y_cat, y_cont


def get_gen_batch(batch_size, cat_dim, cont_dim, noise_dim, noise_scale=0.5):

    X_gen = sample_noise(noise_scale, batch_size, noise_dim)
    y_gen = np.zeros((X_gen.shape[0], 2), dtype=np.uint8)
    y_gen[:, 1] = 1

    y_cat = sample_cat(batch_size, cat_dim)
    y_cont = sample_noise(noise_scale, batch_size, cont_dim)

    # Repeat y_cont to accomodate for keras" loss function conventions
    y_cont_target = np.expand_dims(y_cont, 1)
    y_cont_target = np.repeat(y_cont_target, 2, axis=1)

    return X_gen, y_gen, y_cat, y_cont, y_cont_target


def plot_generated_batch(X_real, generator_model, batch_size, cat_dim, cont_dim, noise_dim, image_data_format, model_name,
                         noise_scale=0.5):

    # Generate images
    y_cat = sample_cat(batch_size, cat_dim)
    y_cont = sample_noise(noise_scale, batch_size, cont_dim)
    noise_input = sample_noise(noise_scale, batch_size, noise_dim)
    # Produce an output
    X_gen = generator_model.predict([y_cat, y_cont, noise_input],batch_size=batch_size)

    X_real = inverse_normalization(X_real)
    X_gen = inverse_normalization(X_gen)

    Xg = X_gen[:8]
    Xr = X_real[:8]

    if image_data_format == "channels_last":
        X = np.concatenate((Xg, Xr), axis=0)
        list_rows = []
        for i in range(int(X.shape[0] / 4)):
            Xr = np.concatenate([X[k] for k in range(4 * i, 4 * (i + 1))], axis=1)
            list_rows.append(Xr)

        Xr = np.concatenate(list_rows, axis=0)

    if image_data_format == "channels_first":
        X = np.concatenate((Xg, Xr), axis=0)
        list_rows = []
        for i in range(int(X.shape[0] / 4)):
            Xr = np.concatenate([X[k] for k in range(4 * i, 4 * (i + 1))], axis=2)
            list_rows.append(Xr)

        Xr = np.concatenate(list_rows, axis=1)
        Xr = Xr.transpose(1,2,0)

    if Xr.shape[-1] == 1:
        plt.imshow(Xr[:, :, 0], cmap="gray")
    else:
        plt.imshow(Xr)
    plt.savefig(os.path.join("../../figures", model_name, "current_batch.png"))
    plt.clf()
    plt.close()
