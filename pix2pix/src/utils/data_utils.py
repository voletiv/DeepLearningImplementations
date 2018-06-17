import cv2
import glob
import h5py
import imageio
import matplotlib.pylab as plt
import numpy as np
import os

from keras.preprocessing.image import ImageDataGenerator


def normalization(X):
    return X / 127.5 - 1


def inverse_normalization(X):
    return np.round((X + 1.) / 2. * 255.).astype('uint8')


def get_nb_patch(img_dim, patch_size, image_data_format):

    assert image_data_format in ["channels_first", "channels_last"], "Bad image_data_format"

    if image_data_format == "channels_first":
        assert img_dim[1] % patch_size[0] == 0, "patch_size does not divide height"
        assert img_dim[2] % patch_size[1] == 0, "patch_size does not divide width"
        nb_patch = (img_dim[1] // patch_size[0]) * (img_dim[2] // patch_size[1])
        img_dim_disc = (img_dim[0], patch_size[0], patch_size[1])

    elif image_data_format == "channels_last":
        assert img_dim[0] % patch_size[0] == 0, "patch_size does not divide height"
        assert img_dim[1] % patch_size[1] == 0, "patch_size does not divide width"
        nb_patch = (img_dim[0] // patch_size[0]) * (img_dim[1] // patch_size[1])
        img_dim_disc = (patch_size[0], patch_size[1], img_dim[-1])

    return nb_patch, img_dim_disc


def extract_patches(X, image_data_format, patch_size):

    # Now extract patches form X_disc
    if image_data_format == "channels_first":
        X = X.transpose(0,2,3,1)

    list_X = []
    list_row_idx = [(i * patch_size[0], (i + 1) * patch_size[0]) for i in range(X.shape[1] // patch_size[0])]
    list_col_idx = [(i * patch_size[1], (i + 1) * patch_size[1]) for i in range(X.shape[2] // patch_size[1])]

    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            list_X.append(X[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])

    if image_data_format == "channels_first":
        for i in range(len(list_X)):
            list_X[i] = list_X[i].transpose(0,3,1,2)

    return list_X


def load_data(dset, image_data_format):

    with h5py.File("../../data/processed/%s_data.h5" % dset, "r") as hf:

        print("Loading train_data_full")
        X_full_train = hf["train_data_full"][:].astype(np.float32)

        print("Loading train_data_sketch")
        X_sketch_train = hf["train_data_sketch"][:].astype(np.float32)

        print("Loading val_data_full")
        X_full_val = hf["val_data_full"][:].astype(np.float32)

        print("Loading val_data_sketch")
        X_sketch_val = hf["val_data_sketch"][:].astype(np.float32)

    X_full_train = normalization(X_full_train)

    X_sketch_train = normalization(X_sketch_train)

    if image_data_format == "channels_last":
        print("transposing dimensions to make channels_last")
        X_full_train = X_full_train.transpose(0, 2, 3, 1)
        X_sketch_train = X_sketch_train.transpose(0, 2, 3, 1)

    X_full_val = normalization(X_full_val)

    X_sketch_val = normalization(X_sketch_val)

    if image_data_format == "channels_last":
        print("transposing dimensions to make channels_last")
        X_full_val = X_full_val.transpose(0, 2, 3, 1)
        X_sketch_val = X_sketch_val.transpose(0, 2, 3, 1)

    return X_full_train, X_sketch_train, X_full_val, X_sketch_val


def data_generator(X_out, X_in, batch_size, augment_data=True):
    
    # data_gen args
    if augment_data:
        data_gen_args = dict(rotation_range=10.,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             horizontal_flip=True)
    else:
        data_gen_args = {}
    
    # datagens
    output_image_datagen = ImageDataGenerator(**data_gen_args)
    input_image_datagen = ImageDataGenerator(**data_gen_args)
    
    # Extend X_out and X_in if their lengths are not a multiple of batch_size
    number_of_elements_to_append = (batch_size - (len(X_out) - (len(X_out)//batch_size * batch_size))) % batch_size
    choices = np.random.choice(len(X_out), number_of_elements_to_append)
    X_out = np.vstack((X_out, X_out[choices]))
    X_in = np.vstack((X_in, X_in[choices]))
    
    # Image generators
    output_image_generator = output_image_datagen.flow(X_out, batch_size=batch_size, seed=29)
    input_image_generator = input_image_datagen.flow(X_in, batch_size=batch_size, seed=29)
    
    return output_image_generator, input_image_generator


def data_generator_from_dir(data_dir, target_size, batch_size):

    # data_gen args
    print("Loading data from", data_dir)

    # Check if number of files in data_dir is a multiple of batch_size
    number_of_images = sum([len(files) for r, d, files in os.walk(data_dir)])
    if number_of_images % batch_size != 0:
        raise ValueError("ERROR: # of images in " + str(data_dir) + " found by keras.ImageDataGenerator is not a multiple of the batch_size ( " + str(batch_size) + " )!\nFound " + str(number_of_images) + " images. Add " + str(batch_size - number_of_images % batch_size) + " more image(s), or delete " + str(number_of_images % batch_size) + " image(s).")

    # datagens
    data_generator_args = {}
    image_datagen = ImageDataGenerator(**data_generator_args)

    # Image generators
    image_data_generator = image_datagen.flow_from_directory(data_dir, target_size=target_size, batch_size=batch_size, class_mode=None, seed=29)

    if len(image_data_generator) == 0:
        raise ValueError("ERROR: # of images found by keras.ImageDataGenerator is 0!\nPlease save the images in the data_dir into at least one modre directory, preferably into classes. Given data_dir:", data_dir)

    return image_data_generator


def load_data_from_data_generator_from_dir(image_data_generator, img_dim=(256, 256), augment_data=True, use_identity_image=False):

    if augment_data:
        data_augmentor_args = dict(rotation_range=10.,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
    else:
        data_augmentor_args = {}

    image_data_aug = ImageDataGenerator(**data_augmentor_args)

    X_batch = next(image_data_generator)
    if use_identity_image:
        # X_batch : target - sketch - identity
        # X_batch_target : target - identity
        # X_batch_sketch : sketch - identity
        X_batch_target = np.concatenate((X_batch[:, :, :img_dim[1]], X_batch[:, :, -img_dim[1]:]), axis=-2)
        X_batch_sketch = X_batch[:, :, img_dim[1]:]
    else:
        X_batch_target = X_batch[:, :, :img_dim[1]]
        X_batch_sketch = X_batch[:, :, img_dim[1]:]

    target_data_augmentor = image_data_aug.flow(X_batch_target, batch_size=len(X_batch_target), shuffle=False, seed=29)
    sketch_data_augmentor = image_data_aug.flow(X_batch_sketch, batch_size=len(X_batch_sketch), shuffle=False, seed=29)

    X_batch_target_aug = next(target_data_augmentor)
    X_batch_sketch_aug = next(sketch_data_augmentor)

    return normalization(X_batch_target_aug), normalization(X_batch_sketch_aug)


def gen_batch(X1, X2, batch_size):

    while True:
        idx = np.random.choice(X1.shape[0], batch_size, replace=False)
        yield X1[idx], X2[idx]


def get_disc_batch(X_full_batch, X_sketch_batch, generator_model, batch_counter, patch_size,
                   image_data_format, label_smoothing=False, label_flipping_prob=0):

    # Create X_disc: alternatively only generated or real images
    if batch_counter % 2 == 0:
        # Produce an output
        X_disc = generator_model.predict(X_sketch_batch)
        y_disc = np.zeros((X_disc.shape[0], 2))
        if label_smoothing:
            y_disc[:, 0] = np.random.uniform(low=0.8, high=1, size=y_disc.shape[0])
            y_disc[:, 1] = np.random.uniform(low=0, high=0.2, size=y_disc.shape[0])
        else:
            y_disc[:, 0] = 1

    else:
        X_disc = X_full_batch
        y_disc = np.zeros((X_disc.shape[0], 2))
        if label_smoothing:
            y_disc[:, 0] = np.random.uniform(low=0, high=0.2, size=y_disc.shape[0])
            y_disc[:, 1] = np.random.uniform(low=0.8, high=1, size=y_disc.shape[0])
        else:
            y_disc[:, 1] = 1

    if np.random.binomial(1, label_flipping_prob) > 0:
        y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    # Now extract patches form X_disc
    X_disc = extract_patches(X_disc, image_data_format, patch_size)

    return X_disc, y_disc


def plot_generated_batch(X_full, X_sketch, generator_model, batch_size, image_data_format, model_name, suffix, iteration_number, MAX_FRAMES_PER_GIF=1000, images_per_row=8):

    # Generate images
    X_gen = generator_model.predict(X_sketch)

    Xs = inverse_normalization(X_sketch[:8])
    Xg = inverse_normalization(X_gen[:8])
    Xf = inverse_normalization(X_full[:8])

    X = np.concatenate((Xs, Xg, Xf), axis=0)
    list_rows = []

    # Make iter text
    text_image = cv2.putText(np.zeros((32, int(Xs[0].shape[1] * images_per_row), Xs[0].shape[2])),
                             'iter %s' % str(iteration_number), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA).astype('uint8')

    if image_data_format == "channels_last":

        for i in range(int(len(X) // images_per_row // 3)):
            list_rows.append(text_image)
            for offset in range(0, len(X), len(X)//3):
                Xr = np.concatenate([X[k] for k in range(images_per_row*i + offset, images_per_row*(i+1) + offset)], axis=1)
                list_rows.append(Xr)

        Xr = np.concatenate(list_rows, axis=0)

    elif image_data_format == "channels_first":

        for i in range(int(len(X) // images_per_row // 3)):
            list_rows.append(text_image)
            for offset in range(0, len(X), len(X)//3):
                Xr = np.concatenate([X[k] for k in range(images_per_row*i + offset, images_per_row*(i+1) + offset)], axis=2)
                list_rows.append(Xr)

        Xr = np.concatenate(list_rows, axis=1)
        Xr = Xr.transpose(1, 2, 0)

    # Save
    imageio.imsave(os.path.join("../../figures", model_name, model_name + "_current_batch_%s.png" % suffix), Xr)

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
                     'iter %s' % str(iteration_number), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA).astype('uint8')
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


def plot_gen_losses(gen_losses, model_name, init_epoch=0):
    epochs = np.arange(len(gen_losses)) + init_epoch
    fig = plt.figure()
    plt.plot(epochs, gen_losses, label='gen_loss')
    plt.legend()
    plt.title("Generator loss")
    plt.xlabel("Epochs")
    plt.savefig(os.path.join("../../figures", model_name, model_name + "_losses.png"))
    plt.clf()
    plt.close()


def plot_losses(disc_losses, gen_total_losses, gen_L1_losses, gen_log_losses, model_name, init_epoch=0):
    # epochs = np.arange(len(disc_losses))
    epochs = np.arange(len(disc_losses)) + init_epoch
    fig = plt.figure()
    plt.subplot(121)
    plt.plot(epochs, disc_losses, label='Discriminator')
    plt.legend()
    plt.title("Discriminator loss")
    plt.xlabel("Epochs")
    # fig.canvas.draw()
    # x_tick_labels = [item.get_text() for item in plt.gca().get_xticklabels()]
    # new_x_tick_labels = []
    # for x_tick_label in x_tick_labels:
    #     try:
    #         new_x_tick_labels.append(int(x_tick_label) + init_epoch)
    #     except:
    #         new_x_tick_labels.append(x_tick_label)
    # plt.gca().set_xticklabels(new_x_tick_labels)
    plt.subplot(122)
    plt.plot(epochs, gen_total_losses, label='Gen_Total')
    plt.plot(epochs, gen_L1_losses, label='Gen_L1')
    plt.plot(epochs, gen_log_losses, label='Gen_log')
    plt.legend()
    plt.title("Generator loss")
    plt.xlabel("Epochs")
    # fig.canvas.draw()
    # x_tick_labels = [item.get_text() for item in plt.gca().get_xticklabels()]
    # new_x_tick_labels = []
    # for x_tick_label in x_tick_labels:
    #     try:
    #         new_x_tick_labels.append(int(x_tick_label) + init_epoch)
    #     except:
    #         new_x_tick_labels.append(x_tick_label)
    # plt.gca().set_xticklabels(new_x_tick_labels)
    plt.savefig(os.path.join("../../figures", model_name, model_name + "_losses.png"))
    plt.clf()
    plt.close()

