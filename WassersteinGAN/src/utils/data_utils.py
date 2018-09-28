import cv2
import glob
import h5py
import imageio
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

from scipy import stats
from keras.datasets import mnist, cifar10
from keras.optimizers import Adam, SGD, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

def normalization(X, image_data_format="channels_last"):

    X = X / 255.
    if image_data_format == "channels_last":
        X = (X - 0.5) / 0.5
    else:
        X = (X - 0.5) / 0.5

    return X


def inverse_normalization(X):

    return ((X * 0.5 + 0.5) * 255.).astype(np.uint8)


def load_mnist(image_data_format):

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if image_data_format == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
        X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    else:
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train = normalization(X_train, image_data_format)
    X_test = normalization(X_test, image_data_format)

    nb_classes = len(np.unique(np.hstack((y_train, y_test))))

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    return X_train, Y_train, X_test, Y_test


def load_cifar10(image_data_format):

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    if image_data_format == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], 3, 32, 32)
        X_test = X_test.reshape(X_test.shape[0], 3, 32, 32)
    else:
        X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
        X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train = normalization(X_train, image_data_format)
    X_test = normalization(X_test, image_data_format)

    nb_classes = len(np.unique(np.vstack((y_train, y_test))))

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    return X_train, Y_train, X_test, Y_test


def load_celebA(img_dim, image_data_format):

    with h5py.File("../../data/processed/CelebA_%s_data.h5" % img_dim, "r") as hf:

        X_real_train = hf["data"][:].astype(np.float32)
        X_real_train = normalization(X_real_train, image_data_format)

        if image_data_format == "channels_last":
            X_real_train = X_real_train.transpose(0, 2, 3, 1)

        return X_real_train


def load_image_dataset(dset, img_dim, image_data_format, batch_size):

    X_batch_gen = None

    if dset == "celebA":
        X_real_train = load_celebA(img_dim, image_data_format)
    elif dset == "mnist":
        X_real_train, _, _, _ = load_mnist(image_data_format)
    elif dset == "cifar10":
        X_real_train, _, _, _ = load_cifar10(image_data_format)
    else:
        X_batch_gen = data_generator_from_dir(dset, (img_dim, img_dim), batch_size, image_data_format)
        X_real_train = next(X_batch_gen)

    return X_real_train, X_batch_gen


def data_generator_from_dir(data_dir, target_size, batch_size, image_data_format="channels_last"):

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


def load_toy(n_mixture=8, std=0.01, radius=1.0, pts_per_mixture=5000):

    thetas = np.linspace(0, 2 * np.pi, n_mixture + 1)[:-1]
    xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
    cov = std * np.eye(2)

    X = np.zeros((n_mixture * pts_per_mixture, 2))

    for i in range(n_mixture):

        mean = np.array([xs[i], ys[i]])
        pts = np.random.multivariate_normal(mean, cov, pts_per_mixture)
        X[i * pts_per_mixture: (i + 1) * pts_per_mixture, :] = pts

    return X


def get_optimizer(opt, lr):

    if opt == "SGD":
        return SGD(lr=lr)
    elif opt == "RMSprop":
        return RMSprop(lr=lr)
    elif opt == "Adam":
        return Adam(lr=lr, beta1=0.5)


def gen_batch(X, X_batch_gen, batch_size):

    while True:
        if X_batch_gen is None:
            idx = np.random.choice(X.shape[0], batch_size, replace=False)
            yield X[idx]
        else:
            yield next(X_batch_gen)


def sample_noise(noise_scale, batch_size, noise_dim):

    return np.random.normal(scale=noise_scale, size=(batch_size, noise_dim[0]))


def get_disc_batch(X_real_batch, generator_model, batch_counter, batch_size, noise_dim, noise_scale=0.5):

    # Pass noise to the generator
    noise_input = sample_noise(noise_scale, batch_size, noise_dim)
    # Produce an output
    X_disc_gen = generator_model.predict(noise_input, batch_size=batch_size)
    X_disc_real = X_real_batch[:batch_size]

    return X_disc_real, X_disc_gen


def save_model_weights(generator_model, discriminator_model, DCGAN_model, e,
                       save_weights_every_n_epochs=5, save_only_last_n_weights=10, model_name="WGAN"):

    purge_weights(generator_model, discriminator_model, DCGAN_model, save_only_last_n_weights, model_name)

    model_path = os.path.join("../../models", model_name)

    if (e + 1) % save_weights_every_n_epochs == 0:
        print("Saving weight...")

        gen_weights_path = os.path.join(model_path, '%s_epoch%5d.h5' % (generator_model.name, e))
        generator_model.save_weights(gen_weights_path, overwrite=True)

        disc_weights_path = os.path.join(model_path, '%s_epoch%5d.h5' % (discriminator_model.name, e))
        discriminator_model.save_weights(disc_weights_path, overwrite=True)

        DCGAN_weights_path = os.path.join(model_path, '%s_epoch%5d.h5' % (DCGAN_model.name, e))
        DCGAN_model.save_weights(DCGAN_weights_path, overwrite=True)


def purge_weights(generator_model, discriminator_model, DCGAN_model, n, model_name):
    gen_weight_files = sorted(glob.glob('../../models/%s/%s*' % (model_name, generator_model.name)))
    for gen_weight_file in gen_weight_files[:-n]:
        os.remove(os.path.realpath(gen_weight_file))

    disc_weight_files = sorted(glob.glob('../../models/%s/%s*' % (model_name, discriminator_model.name)))
    for disc_weight_file in disc_weight_files[:-n]:
        os.remove(os.path.realpath(disc_weight_file))

    DCGAN_weight_files = sorted(glob.glob('../../models/%s/%s*' % (model_name, DCGAN_model.name)))
    for DCGAN_weight_file in DCGAN_weight_files[:-n]:
        os.remove(os.path.realpath(DCGAN_weight_file))


def plot_generated_batch(X_real, generator_model, epoch_number, batch_size,
                         noise_dim, image_data_format, model_name,
                         noise_scale=0.5, suffix='training', MAX_FRAMES_PER_GIF=100):

    # Generate images
    X_gen = sample_noise(noise_scale, batch_size, noise_dim)
    X_gen = generator_model.predict(X_gen)

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

    # Make iter text
    text_image = cv2.putText(np.zeros((32, Xr.shape[1], Xr.shape[2])),
                             '%s epoch' % str(epoch_number), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1, cv2.LINE_AA).astype('uint8')

    image = np.vstack((text_image, Xr))

    # if Xr.shape[-1] == 1:
    #     plt.imshow(Xr[:, :, 0], cmap="gray")
    # else:
    #     plt.imshow(Xr)
    # plt.savefig("../../figures/current_batch.png")
    # plt.clf()
    # plt.close()

    imageio.imsave(os.path.join("../../figures", model_name, model_name + "_current_batch_%s.png" % suffix), image)

    # Make gif
    gif_frames = []

    # Read old gif frames
    try:
        gif_frames_reader = imageio.get_reader(os.path.join("../../figures", model_name, model_name + "_%s.gif" % suffix))
        for frame in gif_frames_reader:
            gif_frames.append(frame[:, :, :3])
    except:
        pass

    # Append new frame
    im = cv2.putText(np.concatenate((np.zeros((32, Xg[0].shape[1], Xg[0].shape[2])), Xg[0]), axis=0),
                     '%s epoch' % str(epoch_number), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1, cv2.LINE_AA).astype('uint8')
    gif_frames.append(im)

    # If frames exceeds, save as different file
    if len(gif_frames) > MAX_FRAMES_PER_GIF:
        print("Splitting the GIF...")
        gif_frames_00 = gif_frames[:MAX_FRAMES_PER_GIF]
        num_of_gifs_already_saved = len(glob.glob(os.path.join("../../figures", model_name, model_name + "_%s_*.gif" % suffix)))
        print("Saving", os.path.join("../../figures", model_name, model_name + "_%s_%03d.gif" % (suffix, num_of_gifs_already_saved)))
        imageio.mimsave(os.path.join("../../figures", model_name, model_name + "_%s_%03d.gif" % (suffix, num_of_gifs_already_saved)), gif_frames_00)
        gif_frames = gif_frames[MAX_FRAMES_PER_GIF:]

    # Save gif
    print("Saving", os.path.join("../../figures", model_name, model_name + "_%s.gif" % suffix))
    imageio.mimsave(os.path.join("../../figures", model_name, model_name + "_%s.gif" % suffix), gif_frames)


def plot_losses(disc_losses, disc_losses_real, disc_losses_gen, gen_losses,
                model_name, init_epoch=0):
    epochs = np.arange(len(disc_losses)) + init_epoch
    fig = plt.figure()
    plt.plot(epochs, disc_losses, linewidth=2, label='D')
    plt.plot(epochs, disc_losses_real, linewidth=1, label='D_real')
    plt.plot(epochs, disc_losses_gen, linewidth=1, label='D_gen')
    plt.plot(epochs, gen_losses, linewidth=2, label='G')
    plt.legend()
    plt.title("Losses")
    plt.xlabel("Epochs")
    plt.savefig(os.path.join("../../figures", model_name, model_name + "_losses.png"), bbox_inches='tight')
    plt.clf()
    plt.close()


def plot_generated_toy_batch(X_real, generator_model, discriminator_model, noise_dim, gen_iter, noise_scale=0.5):

    # Generate images
    X_gen = sample_noise(noise_scale, 10000, noise_dim)
    X_gen = generator_model.predict(X_gen)

    # Get some toy data to plot KDE of real data
    data = load_toy(pts_per_mixture=200)
    x = data[:, 0]
    y = data[:, 1]
    xmin, xmax = -1.5, 1.5
    ymin, ymax = -1.5, 1.5

    # Peform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = stats.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    # Plot the contour
    fig = plt.figure(figsize=(10,10))
    plt.suptitle("Generator iteration %s" % gen_iter, fontweight="bold", fontsize=22)
    ax = fig.gca()
    ax.contourf(xx, yy, f, cmap='Blues', vmin=np.percentile(f,80), vmax=np.max(f), levels=np.linspace(0.25, 0.85, 30))

    # Also plot the contour of the discriminator
    delta = 0.025
    xmin, xmax = -1.5, 1.5
    ymin, ymax = -1.5, 1.5
    # Create mesh
    XX, YY = np.meshgrid(np.arange(xmin, xmax, delta), np.arange(ymin, ymax, delta))
    arr_pos = np.vstack((np.ravel(XX), np.ravel(YY))).T
    # Get Z = predictions
    ZZ = discriminator_model.predict(arr_pos)
    ZZ = ZZ.reshape(XX.shape)
    # Plot contour
    ax.contour(XX, YY, ZZ, cmap="Blues", levels=np.linspace(0.25, 0.85, 10))
    dy, dx = np.gradient(ZZ)
    # Add streamlines
    # plt.streamplot(XX, YY, dx, dy, linewidth=0.5, cmap="magma", density=1, arrowsize=1)
    # Scatter generated data
    plt.scatter(X_gen[:1000, 0], X_gen[:1000, 1], s=20, color="coral", marker="o")

    l_gen = plt.Line2D((0,1),(0,0), color='coral', marker='o', linestyle='', markersize=20)
    l_D = plt.Line2D((0,1),(0,0), color='steelblue', linewidth=3)
    l_real = plt.Rectangle((0, 0), 1, 1, fc="steelblue")

    # Create legend from custom artist/label lists
    # bbox_to_anchor = (0.4, 1)
    ax.legend([l_real, l_D, l_gen], ['Real data KDE', 'Discriminator contour',
                                     'Generated data'], fontsize=18, loc="upper left")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax + 0.8)
    plt.savefig("../../figures/toy_dataset_iter%s.jpg" % gen_iter)
    plt.clf()
    plt.close()


if __name__ == '__main__':

    data = load_toy(pts_per_mixture=200)

    x = data[:, 0]
    y = data[:, 1]
    xmin, xmax = -1.5, 1.5
    ymin, ymax = -1.5, 1.5

    # Peform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = stats.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    fig = plt.figure()
    gen_it = 5
    plt.suptitle("Generator iteration %s" % gen_it, fontweight="bold")
    ax = fig.gca()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # Contourf plot
    cfset = ax.contourf(xx, yy, f, cmap='Blues', vmin=np.percentile(f,90),
                        vmax=np.max(f), levels=np.linspace(0.25, 0.85, 30))
    # cfset = ax.contour(xx, yy, f, color="k", levels=np.linspace(0.25, 0.85, 30), label="roger")
    plt.legend()
    plt.show()
