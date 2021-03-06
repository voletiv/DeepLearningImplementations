import datetime
import glob
import imageio
import json
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


def purge_weights(n, model_name):
    # disc
    disc_weight_files = glob.glob('../../models/%s/disc_weights*' % model_name)
    for disc_weight_file in disc_weight_files[:-n]:
        os.remove(os.path.realpath(disc_weight_file))
    # gen
    gen_weight_files = glob.glob('../../models/%s/gen_weights*' % model_name)
    for gen_weight_file in gen_weight_files[:-n]:
        os.remove(os.path.realpath(gen_weight_file))
    # DCGAN
    DCGAN_weight_files = glob.glob('../../models/%s/DCGAN_weights*' % model_name)
    for DCGAN_weight_file in DCGAN_weight_files[:-n]:
        os.remove(os.path.realpath(DCGAN_weight_file))


def train(**kwargs):
    """
    Train model

    Load the whole train data in memory for faster operations

    args: **kwargs (dict) keyword arguments that specify the model hyperparameters
    """

    # Roll out the parameters
    img_dim = kwargs["img_dim"]
    patch_size = kwargs["patch_size"]
    image_data_format = kwargs["image_data_format"]
    generator_type = kwargs["generator_type"]
    dset = kwargs["dset"]
    use_identity_image = kwargs["use_identity_image"]
    batch_size = kwargs["batch_size"]
    n_batch_per_epoch = kwargs["n_batch_per_epoch"]
    nb_epoch = kwargs["nb_epoch"]
    augment_data = kwargs["augment_data"]
    model_name = kwargs["model_name"]
    save_weights_every_n_epochs = kwargs["save_weights_every_n_epochs"]
    visualize_images_every_n_epochs = kwargs["visualize_images_every_n_epochs"]
    save_only_last_n_weights = kwargs["save_only_last_n_weights"]
    use_mbd = kwargs["use_mbd"]
    label_smoothing = kwargs["use_label_smoothing"]
    label_flipping_prob = kwargs["label_flipping_prob"]
    use_l1_weighted_loss = kwargs["use_l1_weighted_loss"]
    prev_model = kwargs["prev_model"]
    change_model_name_to_prev_model = kwargs["change_model_name_to_prev_model"]
    discriminator_optimizer = kwargs["discriminator_optimizer"]
    n_run_of_gen_for_1_run_of_disc = kwargs["n_run_of_gen_for_1_run_of_disc"]
    load_all_data_at_once = kwargs["load_all_data_at_once"]
    MAX_FRAMES_PER_GIF = kwargs["MAX_FRAMES_PER_GIF"]
    dont_train = kwargs["dont_train"]

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

    if dont_train:
        # Get the number of non overlapping patch and the size of input image to the discriminator
        nb_patch, img_dim_disc = data_utils.get_nb_patch(img_dim, patch_size, image_data_format)
        generator_model = models.load("generator_unet_%s" % generator_type,
                                      img_dim,
                                      nb_patch,
                                      use_mbd,
                                      batch_size,
                                      model_name)
        generator_model.compile(loss='mae', optimizer='adam')
        return generator_model

    # Check and make the dataset
    # If .h5 file of dset is not present, try making it
    if load_all_data_at_once:
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
    else:
        if not os.path.exists(dset):
            print("dset does not exist! Given:", dset)
            return
        if not os.path.exists(os.path.join(dset, 'train')):
            print("dset does not contain a 'train' dir! Given dset:", dset)
            return
        if not os.path.exists(os.path.join(dset, 'val')):
            print("dset does not contain a 'val' dir! Given dset:", dset)
            return

    epoch_size = n_batch_per_epoch * batch_size

    init_epoch = 0

    if prev_model:
        print('\n\nLoading prev_model from', prev_model, '...\n\n')
        prev_model_latest_gen = sorted(glob.glob(os.path.join('../../models/', prev_model, '*gen*epoch*.h5')))[-1]
        prev_model_latest_disc = sorted(glob.glob(os.path.join('../../models/', prev_model, '*disc*epoch*.h5')))[-1]
        prev_model_latest_DCGAN = sorted(glob.glob(os.path.join('../../models/', prev_model, '*DCGAN*epoch*.h5')))[-1]
        print(prev_model_latest_gen, prev_model_latest_disc, prev_model_latest_DCGAN)
        if change_model_name_to_prev_model:
            # Find prev model name, epoch
            model_name = prev_model_latest_DCGAN.split('models')[-1].split('/')[1]
            init_epoch = int(prev_model_latest_DCGAN.split('epoch')[1][:5]) + 1

    # img_dim = X_target_train.shape[-3:]
    # img_dim = (256, 256, 3)

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

        generator_model.compile(loss='mae', optimizer=opt_dcgan)

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

        if use_l1_weighted_loss:
            loss = [l1_weighted_loss, 'binary_crossentropy']
        else:
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

        # Load .h5 data all at once
        print('\n\nLoading data...\n\n')
        check_this_process_memory()

        if load_all_data_at_once:
            X_target_train, X_sketch_train, X_target_val, X_sketch_val = data_utils.load_data(dset, image_data_format)
            check_this_process_memory()
            print('X_target_train: %.4f' % (X_target_train.nbytes/2**30), "GB")
            print('X_sketch_train: %.4f' % (X_sketch_train.nbytes/2**30), "GB")
            print('X_target_val: %.4f' % (X_target_val.nbytes/2**30), "GB")
            print('X_sketch_val: %.4f' % (X_sketch_val.nbytes/2**30), "GB")

            # To generate training data
            X_target_batch_gen_train, X_sketch_batch_gen_train = data_utils.data_generator(X_target_train, X_sketch_train, batch_size, augment_data=augment_data)
            X_target_batch_gen_val, X_sketch_batch_gen_val = data_utils.data_generator(X_target_val, X_sketch_val, batch_size, augment_data=False)

        # Load data from images through an ImageDataGenerator
        else:
            X_batch_gen_train = data_utils.data_generator_from_dir(os.path.join(dset, 'train'), target_size=(img_dim[0], 2*img_dim[1]), batch_size=batch_size)
            X_batch_gen_val = data_utils.data_generator_from_dir(os.path.join(dset, 'val'), target_size=(img_dim[0], 2*img_dim[1]), batch_size=batch_size)

        check_this_process_memory()

        # Setup environment (logging directory etc)
        general_utils.setup_logging(**kwargs)

        # Losses
        disc_losses = []
        gen_total_losses = []
        gen_L1_losses = []
        gen_log_losses = []

        # Start training
        print("\n\nStarting training...\n\n")

        # For each epoch
        for e in range(nb_epoch):
            
            # Initialize progbar and batch counter
            # progbar = generic_utils.Progbar(epoch_size)
            batch_counter = 0
            gen_total_loss_epoch = 0
            gen_L1_loss_epoch = 0
            gen_log_loss_epoch = 0
            start = time.time()
            
            # For each batch
            # for X_target_batch, X_sketch_batch in data_utils.gen_batch(X_target_train, X_sketch_train, batch_size):
            for batch in range(n_batch_per_epoch):
                
                # Create a batch to feed the discriminator model
                if load_all_data_at_once:
                    X_target_batch_train, X_sketch_batch_train = next(X_target_batch_gen_train), next(X_sketch_batch_gen_train)
                else:
                    X_target_batch_train, X_sketch_batch_train = data_utils.load_data_from_data_generator_from_dir(X_batch_gen_train, img_dim=img_dim,
                                                                                                                   augment_data=augment_data,
                                                                                                                   use_identity_image=use_identity_image)

                X_disc, y_disc = data_utils.get_disc_batch(X_target_batch_train,
                                                           X_sketch_batch_train,
                                                           generator_model,
                                                           batch_counter,
                                                           patch_size,
                                                           image_data_format,
                                                           label_smoothing=label_smoothing,
                                                           label_flipping_prob=label_flipping_prob)
                
                # Update the discriminator
                disc_loss = discriminator_model.train_on_batch(X_disc, y_disc)
                
                # Create a batch to feed the generator model
                if load_all_data_at_once:
                    X_gen_target, X_gen_sketch = next(X_target_batch_gen_train), next(X_sketch_batch_gen_train)
                else:
                    X_gen_target, X_gen_sketch = data_utils.load_data_from_data_generator_from_dir(X_batch_gen_train, img_dim=img_dim,
                                                                                                   augment_data=augment_data,
                                                                                                   use_identity_image=use_identity_image)

                y_gen_target = np.zeros((X_gen_target.shape[0], 2), dtype=np.uint8)
                y_gen_target[:, 1] = 1
                
                # Freeze the discriminator
                discriminator_model.trainable = False
                
                # Train generator
                for _ in range(n_run_of_gen_for_1_run_of_disc-1):
                    gen_loss = DCGAN_model.train_on_batch(X_gen_sketch, [X_gen_target, y_gen_target])
                    gen_total_loss_epoch += gen_loss[0]/n_run_of_gen_for_1_run_of_disc
                    gen_L1_loss_epoch += gen_loss[1]/n_run_of_gen_for_1_run_of_disc
                    gen_log_loss_epoch += gen_loss[2]/n_run_of_gen_for_1_run_of_disc
                    if load_all_data_at_once:
                        X_gen_target, X_gen_sketch = next(X_target_batch_gen_train), next(X_sketch_batch_gen_train)
                    else:
                        X_gen_target, X_gen_sketch = data_utils.load_data_from_data_generator_from_dir(X_batch_gen_train, img_dim=img_dim,
                                                                                                       augment_data=augment_data,
                                                                                                       use_identity_image=use_identity_image)

                gen_loss = DCGAN_model.train_on_batch(X_gen_sketch, [X_gen_target, y_gen_target])
                
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
                print("Epoch", str(init_epoch+e+1), "batch", str(batch+1), "D_logloss", disc_loss, "G_tot", gen_loss[0], "G_L1", gen_loss[1], "G_log", gen_loss[2])
            
            gen_total_loss = gen_total_loss_epoch/n_batch_per_epoch
            gen_L1_loss = gen_L1_loss_epoch/n_batch_per_epoch
            gen_log_loss = gen_log_loss_epoch/n_batch_per_epoch
            disc_losses.append(disc_loss)
            gen_total_losses.append(gen_total_loss)
            gen_L1_losses.append(gen_L1_loss)
            gen_log_losses.append(gen_log_loss)
            
            # Save images for visualization
            if (e + 1) % visualize_images_every_n_epochs == 0:
                data_utils.plot_generated_batch(X_target_batch_train, X_sketch_batch_train, generator_model, batch_size, image_data_format,
                                                model_name, "training", init_epoch + e + 1, MAX_FRAMES_PER_GIF)
                # Get new images for validation
                if load_all_data_at_once:
                    X_target_batch_val, X_sketch_batch_val = next(X_target_batch_gen_val), next(X_sketch_batch_gen_val)
                else:
                    X_target_batch_val, X_sketch_batch_val = data_utils.load_data_from_data_generator_from_dir(X_batch_gen_val, img_dim=img_dim, augment_data=False, use_identity_image=use_identity_image)
                # Predict and validate
                data_utils.plot_generated_batch(X_target_batch_val, X_sketch_batch_val, generator_model, batch_size, image_data_format,
                                                model_name, "validation", init_epoch + e + 1, MAX_FRAMES_PER_GIF)
                # Plot losses
                data_utils.plot_losses(disc_losses, gen_total_losses, gen_L1_losses, gen_log_losses, model_name, init_epoch)
            
            # Save weights
            if (e + 1) % save_weights_every_n_epochs == 0:
                # Delete all but the last n weights
                purge_weights(save_only_last_n_weights, model_name)
                # Save gen weights
                gen_weights_path = os.path.join('../../models/%s/gen_weights_epoch%05d_discLoss%.04f_genTotL%.04f_genL1L%.04f_genLogL%.04f.h5' % (model_name, init_epoch + e, disc_losses[-1], gen_total_losses[-1], gen_L1_losses[-1], gen_log_losses[-1]))
                print("Saving", gen_weights_path)
                generator_model.save_weights(gen_weights_path, overwrite=True)
                # Save disc weights
                disc_weights_path = os.path.join('../../models/%s/disc_weights_epoch%05d_discLoss%.04f_genTotL%.04f_genL1L%.04f_genLogL%.04f.h5' % (model_name, init_epoch + e, disc_losses[-1], gen_total_losses[-1], gen_L1_losses[-1], gen_log_losses[-1]))
                print("Saving", disc_weights_path)
                discriminator_model.save_weights(disc_weights_path, overwrite=True)
                # Save DCGAN weights
                DCGAN_weights_path = os.path.join('../../models/%s/DCGAN_weights_epoch%05d_discLoss%.04f_genTotL%.04f_genL1L%.04f_genLogL%.04f.h5' % (model_name, init_epoch + e, disc_losses[-1], gen_total_losses[-1], gen_L1_losses[-1], gen_log_losses[-1]))
                print("Saving", DCGAN_weights_path)
                DCGAN_model.save_weights(DCGAN_weights_path, overwrite=True)

            check_this_process_memory()
            print('[{0:%Y/%m/%d %H:%M:%S}] Epoch {1:d}/{2:d} END, Time taken: {3:.4f} seconds'.format(datetime.datetime.now(), init_epoch + e + 1, init_epoch + nb_epoch, time.time() - start))
            print('------------------------------------------------------------------------------------')

    except KeyboardInterrupt:
        pass

    # SAVE THE MODEL

    try:
        # Save the model as it is, so that it can be loaded using -
        # ```from keras.models import load_model; gen = load_model('generator_latest.h5')```
        gen_weights_path = '../../models/%s/generator_latest.h5' % (model_name)
        print("Saving", gen_weights_path)
        generator_model.save(gen_weights_path, overwrite=True)
    
        # Save model as json string
        generator_model_json_string = generator_model.to_json()
        print("Saving", '../../models/%s/generator_latest.txt' % model_name)
        with open('../../models/%s/generator_latest.txt' % model_name, 'w') as outfile:
            a = outfile.write(generator_model_json_string)
    
        # Save model as json
        generator_model_json_data = json.loads(generator_model_json_string)
        print("Saving", '../../models/%s/generator_latest.json' % model_name)
        with open('../../models/%s/generator_latest.json' % model_name, 'w') as outfile:
            json.dump(generator_model_json_data, outfile)

    except:
        print(sys.exc_info()[0])

    print("Done.")

    return generator_model

