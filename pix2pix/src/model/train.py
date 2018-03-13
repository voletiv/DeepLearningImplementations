import glob
import imageio
import numpy as np
import os
import psutil
import subprocess
import sys
import time

import models

import tensorflow as tf

import keras.backend as K
from keras.utils import generic_utils
from keras.optimizers import Adam, SGD

# Utils
sys.path.append("../utils")
import general_utils
import data_utils

w = 256
a = np.exp(-np.linspace(-w//2+1, w//2, w)**2/(50/(w/150))**2)
b = np.exp(-np.linspace(-w//2+1, w//2, w)**2/(50/(w/200))**2)
gaussian_overlap = a[:, np.newaxis] * b[np.newaxis, :]
gaussian_overlap = np.vstack((np.zeros((30, w)), gaussian_overlap[:-30]))
gaussian_overlap = tf.convert_to_tensor(gaussian_overlap, dtype=tf.float32)


def l1_weighted_loss(y_true, y_pred):
    reconstruction_loss = K.sum(K.abs(y_pred - y_true), axis=-1)
    return reconstruction_loss + tf.multiply(reconstruction_loss, gaussian_overlap)


def l1_loss(y_true, y_pred):
    return K.sum(K.abs(y_pred - y_true), axis=-1)


def check_this_process_memory():
    memoryUse = psutil.Process(os.getpid()).memory_info()[0]/2.**30  # memory use in GB
    print('memory use: %.4f' % memoryUse, 'GB')


def train(**kwargs):
    """
    Train model

    Load the whole train data in memory for faster operations

    args: **kwargs (dict) keyword arguments that specify the model hyperparameters
    """

    # Roll out the parameters
    patch_size = kwargs["patch_size"]
    image_data_format = kwargs["image_data_format"]
    generator_type = kwargs["generator_type"]
    dset = kwargs["dset"]
    batch_size = kwargs["batch_size"]
    n_batch_per_epoch = kwargs["n_batch_per_epoch"]
    nb_epoch = kwargs["nb_epoch"]
    model_name = kwargs["model_name"]
    save_weights_every_n_epochs = kwargs["save_weights_every_n_epochs"]
    visualize_images_every_n_epochs = kwargs["visualize_images_every_n_epochs"]
    use_mbd = kwargs["use_mbd"]
    label_smoothing = kwargs["use_label_smoothing"]
    label_flipping_prob = kwargs["label_flipping_prob"]
    use_l1_weighted_loss = kwargs["use_l1_weighted_loss"]
    prev_model = kwargs["prev_model"]
    discriminator_optimizer = kwargs["discriminator_optimizer"]
    n_run_of_gen_for_1_run_of_disc = kwargs["n_run_of_gen_for_1_run_of_disc"]
    MAX_FRAMES_PER_GIF = kwargs["MAX_FRAMES_PER_GIF"]

    # batch_size = args.batch_size
    # n_batch_per_epoch = args.n_batch_per_epoch
    # nb_epoch = args.nb_epoch
    # save_weights_every_n_epochs = args.save_weights_every_n_epochs
    # generator_type = args.generator_type
    # patch_size = args.patch_size
    # label_smoothing = False
    # label_flipping_prob = False
    # dset = args.dset
    # use_mbd = False

    # Check and make the dataset
    # If .h5 file of dset is not present, try making it
    if not os.path.exists("../../data/processed/%s_data.h5" % dset):
        print("dset %s_data.h5 not present in '../../data/processed'!" % dset)
        if not os.path.exists("../../data/%s/" % dset):
            print("dset folder %s not present in '../../data'!\n\nERROR: Dataset .h5 file not made, and dataset not available in '../../data/'.\n\nQuitting." % dset)
            return
        else:
            if not os.path.exists("../../data/%s/train" % dset) or not os.path.exists("../../data/%s/val" % dset) or not os.path.exists("../../data/%s/test" % dset):
                print("'train', 'val' or 'test' folders not present in dset folder '../../data/%s'!\n\nERROR: Dataset must contain 'train', 'val' and 'test' folders.\n\nQuitting." % dset)
                return
            else:
                print("Making %s dataset" % dset)
                subprocess.call(['python3', '../data/make_dataset.py', '../../data/%s' % dset, '3'])
                print("Done!")

    epoch_size = n_batch_per_epoch * batch_size

    init_epoch = 0

    if prev_model:
        print('\n\nLoading prev_model from', prev_model, '...\n\n')
        prev_model_latest_gen = sorted(glob.glob(os.path.join('../../models/', prev_model, '*gen*.h5')))[-1]
        prev_model_latest_disc = sorted(glob.glob(os.path.join('../../models/', prev_model, '*disc*.h5')))[-1]
        prev_model_latest_DCGAN = sorted(glob.glob(os.path.join('../../models/', prev_model, '*DCGAN*.h5')))[-1]
        # Find prev model name, epoch
        model_name = prev_model_latest_DCGAN.split('models')[-1].split('/')[1]
        init_epoch = int(prev_model_latest_DCGAN.split('epoch')[1][:5]) + 1

    # Setup environment (logging directory etc), if no prev_model is mentioned
    general_utils.setup_logging(model_name)

    # img_dim = X_full_train.shape[-3:]
    img_dim = (256, 256, 3)

    # Get the number of non overlapping patch and the size of input image to the discriminator
    nb_patch, img_dim_disc = data_utils.get_nb_patch(img_dim, patch_size, image_data_format)

    try:

        # Create optimizers
        opt_dcgan = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        if discriminator_optimizer == 'sgd':
            opt_discriminator = SGD(lr=1E-3, momentum=0.9, nesterov=True)
        elif discriminator_optimizer == 'adam':
            opt_discriminator = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        # Load generator model
        generator_model = models.load("generator_unet_%s" % generator_type,
                                      img_dim,
                                      nb_patch,
                                      use_mbd,
                                      batch_size,
                                      model_name)

        generator_model.compile(loss='mae', optimizer=opt_discriminator)

        # Load discriminator model
        discriminator_model = models.load("DCGAN_discriminator",
                                          img_dim_disc,
                                          nb_patch,
                                          use_mbd,
                                          batch_size,
                                          model_name)

        discriminator_model.trainable = False

        DCGAN_model = models.DCGAN(generator_model,
                                   discriminator_model,
                                   img_dim,
                                   patch_size,
                                   image_data_format)

        loss = [l1_loss, 'binary_crossentropy']
        loss_weights = [1E1, 1]
        DCGAN_model.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)

        discriminator_model.trainable = True
        discriminator_model.compile(loss='binary_crossentropy', optimizer=opt_discriminator)

        # Load prev_model
        if prev_model:
            generator_model.load_weights(prev_model_latest_gen)
            discriminator_model.load_weights(prev_model_latest_disc)
            DCGAN_model.load_weights(prev_model_latest_DCGAN)

        # Load and rescale data
        print('\n\nLoading data...\n\n')
        X_full_train, X_sketch_train, X_full_val, X_sketch_val = data_utils.load_data(dset, image_data_format)
        check_this_process_memory()
        print('X_full_train: %.4f' % (X_full_train.nbytes/2**30), "GB")
        print('X_sketch_train: %.4f' % (X_sketch_train.nbytes/2**30), "GB")
        print('X_full_val: %.4f' % (X_full_val.nbytes/2**30), "GB")
        print('X_sketch_val: %.4f' % (X_sketch_val.nbytes/2**30), "GB")

        # Losses
        disc_losses = []
        gen_total_losses = []
        gen_L1_losses = []
        gen_log_losses = []

        # Start training
        print("\n\nStarting training\n\n")
        for e in range(nb_epoch):
            # Initialize progbar and batch counter
            # progbar = generic_utils.Progbar(epoch_size)
            batch_counter = 0
            gen_total_loss_epoch = 0
            gen_L1_loss_epoch = 0
            gen_log_loss_epoch = 0
            start = time.time()
            for X_full_batch, X_sketch_batch in data_utils.gen_batch(X_full_train, X_sketch_train, batch_size):
                # Create a batch to feed the discriminator model
                X_disc, y_disc = data_utils.get_disc_batch(X_full_batch,
                                                           X_sketch_batch,
                                                           generator_model,
                                                           batch_counter,
                                                           patch_size,
                                                           image_data_format,
                                                           label_smoothing=label_smoothing,
                                                           label_flipping_prob=label_flipping_prob)
                # Update the discriminator
                disc_loss = discriminator_model.train_on_batch(X_disc, y_disc)
                # Create a batch to feed the generator model
                X_gen_target, X_gen = next(data_utils.gen_batch(X_full_train, X_sketch_train, batch_size))
                y_gen = np.zeros((X_gen.shape[0], 2), dtype=np.uint8)
                y_gen[:, 1] = 1
                # Freeze the discriminator
                discriminator_model.trainable = False
                # Train generator
                for _ in range(n_run_of_gen_for_1_run_of_disc-1):
                    gen_loss = DCGAN_model.train_on_batch(X_gen, [X_gen_target, y_gen])
                    gen_total_loss_epoch += gen_loss[0]/n_run_of_gen_for_1_run_of_disc
                    gen_L1_loss_epoch += gen_loss[1]/n_run_of_gen_for_1_run_of_disc
                    gen_log_loss_epoch += gen_loss[2]/n_run_of_gen_for_1_run_of_disc
                    X_gen_target, X_gen = next(data_utils.gen_batch(X_full_train, X_sketch_train, batch_size))
                gen_loss = DCGAN_model.train_on_batch(X_gen, [X_gen_target, y_gen])
                # Add losses
                gen_total_loss_epoch += gen_loss[0]/n_run_of_gen_for_1_run_of_disc
                gen_L1_loss_epoch += gen_loss[1]/n_run_of_gen_for_1_run_of_disc
                gen_log_loss_epoch += gen_loss[2]/n_run_of_gen_for_1_run_of_disc
                # Unfreeze the discriminator
                discriminator_model.trainable = True
                # Progress
                # progbar.add(batch_size, values=[("D logloss", disc_loss),
                #                                 ("G tot", gen_loss[0]),
                #                                 ("G L1", gen_loss[1]),
                #                                 ("G logloss", gen_loss[2])])
                print("Epoch", str(init_epoch+e+1), "batch", str(batch_counter+1), "D_logloss", disc_loss, "G_tot", gen_loss[0], "G_L1", gen_loss[1], "G_log", gen_loss[2])
                batch_counter += 1
                if batch_counter >= n_batch_per_epoch:
                    break
            gen_total_loss = gen_total_loss_epoch/n_batch_per_epoch
            gen_L1_loss = gen_L1_loss_epoch/n_batch_per_epoch
            gen_log_loss = gen_log_loss_epoch/n_batch_per_epoch
            disc_losses.append(disc_loss)
            gen_total_losses.append(gen_total_loss)
            gen_L1_losses.append(gen_L1_loss)
            gen_log_losses.append(gen_log_loss)
            check_this_process_memory()
            print('Epoch %s/%s, Time: %.4f' % (init_epoch + e + 1, init_epoch + nb_epoch, time.time() - start))
            # Save images for visualization
            if (e + 1) % visualize_images_every_n_epochs == 0:
                data_utils.plot_generated_batch(X_full_batch, X_sketch_batch, generator_model, batch_size, image_data_format,
                                                model_name, "training", init_epoch + e + 1, MAX_FRAMES_PER_GIF)
                # Get new images from validation
                X_full_batch, X_sketch_batch = next(data_utils.gen_batch(X_full_val, X_sketch_val, batch_size))
                data_utils.plot_generated_batch(X_full_batch, X_sketch_batch, generator_model, batch_size, image_data_format,
                                                model_name, "validation", init_epoch + e + 1, MAX_FRAMES_PER_GIF)
                # Plot losses
                data_utils.plot_losses(disc_losses, gen_total_losses, gen_L1_losses, gen_log_losses, model_name, init_epoch)
            # Save weights
            if (e + 1) % save_weights_every_n_epochs == 0:
                gen_weights_path = os.path.join('../../models/%s/gen_weights_epoch%05d_discLoss%.04f_genTotL%.04f_genL1L%.04f_genLogL%.04f.h5' % (model_name, init_epoch + e, disc_losses[-1], gen_total_losses[-1], gen_L1_losses[-1], gen_log_losses[-1]))
                generator_model.save_weights(gen_weights_path, overwrite=True)
                disc_weights_path = os.path.join('../../models/%s/disc_weights_epoch%05d_discLoss%.04f_genTotL%.04f_genL1L%.04f_genLogL%.04f.h5' % (model_name, init_epoch + e, disc_losses[-1], gen_total_losses[-1], gen_L1_losses[-1], gen_log_losses[-1]))
                discriminator_model.save_weights(disc_weights_path, overwrite=True)
                DCGAN_weights_path = os.path.join('../../models/%s/DCGAN_weights_epoch%05d_discLoss%.04f_genTotL%.04f_genL1L%.04f_genLogL%.04f.h5' % (model_name, init_epoch + e, disc_losses[-1], gen_total_losses[-1], gen_L1_losses[-1], gen_log_losses[-1]))
                DCGAN_model.save_weights(DCGAN_weights_path, overwrite=True)

    except KeyboardInterrupt:
        pass
