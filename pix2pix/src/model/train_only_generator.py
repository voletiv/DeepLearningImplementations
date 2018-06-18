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
gaussian_overlap_tf = tf.convert_to_tensor(gaussian_overlap, dtype=tf.float32)
gaussian_overlap_left_tf = tf.convert_to_tensor(np.hstack((gaussian_overlap, np.zeros((w, w)))), dtype=tf.float32)
gaussian_overlap_right_tf = tf.convert_to_tensor(np.hstack((np.zeros((w, w)), gaussian_overlap)), dtype=tf.float32)


def l1_weighted_identity_loss(y_true, y_pred):
    l1_loss = K.mean(K.abs(y_pred - y_true), axis=-1)
    return l1_loss + tf.multiply(l1_loss, gaussian_overlap_left_tf) + tf.multiply(l1_loss, gaussian_overlap_right_tf)


def l1_weighted_loss(y_true, y_pred):
    reconstruction_loss = K.mean(K.abs(y_pred - y_true), axis=-1)
    return reconstruction_loss + tf.multiply(reconstruction_loss, gaussian_overlap_tf)


def l1_loss(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)


def check_this_process_memory():
    memoryUse = psutil.Process(os.getpid()).memory_info()[0]/2.**30  # memory use in GB
    print('memory use: %.4f' % memoryUse, 'GB')


def purge_weights(n, model_name):
    gen_weight_files = glob.glob('../../models/%s/gen_weights*' % model_name)
    for gen_weight_file in gen_weight_files[:-n]:
        os.remove(os.path.realpath(gen_weight_file))


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
        print(prev_model_latest_gen)
        # Find prev model name, epoch
        if change_model_name_to_prev_model:
            model_name = prev_model_latest_gen.split('models')[-1].split('/')[1]
            init_epoch = int(prev_model_latest_gen.split('epoch')[1][:5]) + 1

    # img_dim = X_target_train.shape[-3:]
    # img_dim = (256, 256, 3)

    # Get the number of non overlapping patch and the size of input image to the discriminator
    nb_patch, img_dim_disc = data_utils.get_nb_patch(img_dim, patch_size, image_data_format)

    if use_identity_image:
        gen_input_img_dim = [img_dim[0], 2*img_dim[1], img_dim[2]]
    else:
        gen_input_img_dim = img_dim

    try:

        # Create optimizer
        opt_generator = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        # Load generator model
        generator_model = models.load("generator_unet_%s" % generator_type,
                                      gen_input_img_dim,
                                      nb_patch,
                                      use_mbd,
                                      batch_size,
                                      model_name)

        if use_l1_weighted_loss and use_identity_image:
            loss = l1_weighted_identity_loss
        elif use_l1_weighted_loss and not use_identity_image:
            loss = l1_weighted_loss
        else:
            loss = l1_loss

        generator_model.compile(loss=loss, optimizer=opt_generator)

        # Load prev_model
        if prev_model:
            generator_model.load_weights(prev_model_latest_gen)

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
            if use_identity_image:
                X_batch_gen_train = data_utils.data_generator_from_dir(os.path.join(dset, 'train'), target_size=(img_dim[0], 3*img_dim[1]), batch_size=batch_size)
                X_batch_gen_val = data_utils.data_generator_from_dir(os.path.join(dset, 'val'), target_size=(img_dim[0], 3*img_dim[1]), batch_size=batch_size)
            else:
                X_batch_gen_train = data_utils.data_generator_from_dir(os.path.join(dset, 'train'), target_size=(img_dim[0], 2*img_dim[1]), batch_size=batch_size)
                X_batch_gen_val = data_utils.data_generator_from_dir(os.path.join(dset, 'val'), target_size=(img_dim[0], 2*img_dim[1]), batch_size=batch_size)

        check_this_process_memory()

        if dont_train:
            raise KeyboardInterrupt

        # Setup environment (logging directory etc)
        general_utils.setup_logging(**kwargs)

        # Losses
        gen_losses = []

        # Start training
        print("\n\nStarting training...\n\n")

        # For each epoch
        for e in range(nb_epoch):
            
            # Initialize progbar and batch counter
            # progbar = generic_utils.Progbar(epoch_size)
            batch_counter = 0
            gen_loss_epoch = 0
            start = time.time()
            
            # For each batch
            # for X_target_batch, X_sketch_batch in data_utils.gen_batch(X_target_train, X_sketch_train, batch_size):
            for batch in range(n_batch_per_epoch):
                
                # Create a batch to feed the generator model
                if load_all_data_at_once:
                    X_gen_target, X_gen_sketch = next(X_target_batch_gen_train), next(X_sketch_batch_gen_train)
                else:
                    X_gen_target, X_gen_sketch = data_utils.load_data_from_data_generator_from_dir(X_batch_gen_train, img_dim=img_dim, augment_data=augment_data, use_identity_image=use_identity_image)
                
                # Train generator
                gen_loss = generator_model.train_on_batch(X_gen_sketch, X_gen_target)
                
                # Add losses
                gen_loss_epoch += gen_loss
                
                print("Epoch", str(init_epoch+e+1), "batch", str(batch+1), "G_loss", gen_loss)
            
            # Append loss
            gen_losses.append(gen_loss_epoch/n_batch_per_epoch)
            
            # Save images for visualization
            if (e + 1) % visualize_images_every_n_epochs == 0:
                data_utils.plot_generated_batch(X_gen_target, X_gen_sketch, generator_model, batch_size, image_data_format,
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
                data_utils.plot_gen_losses(gen_losses, model_name, init_epoch)
            
            # Save weights
            if (e + 1) % save_weights_every_n_epochs == 0:
                # Delete all but the last n weights
                purge_weights(save_only_last_n_weights, model_name)
                # Save gen weights
                gen_weights_path = os.path.join('../../models/%s/gen_weights_epoch%05d_genLoss%.04f.h5' % (model_name, init_epoch + e, gen_losses[-1]))
                print("Saving", gen_weights_path)
                generator_model.save_weights(gen_weights_path, overwrite=True)

            check_this_process_memory()
            print('[{0:%Y/%m/%d %H:%M:%S}] Epoch {1:d}/{2:d} END, Time taken: {3:.4f} seconds'.format(datetime.datetime.now(), init_epoch + e + 1, init_epoch + nb_epoch, time.time() - start))
            print('------------------------------------------------------------------------------------')

    except KeyboardInterrupt:
        if dont_train:
            return generator_model
        else:
            pass

    # SAVE THE MODEL

    # Save the model as it is, so that it can be loaded using -
    # ```from keras.models import load_model; gen = load_model('generator_latest.h5')```
    gen_weights_path = '../../models/%s/generator_latest.h5' % (model_name)
    print("Saving", gen_weights_path)
    if use_l1_weighted_loss:
        generator_model.compile(loss='mae', optimizer=opt_generator)
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

    print("Done.")

    return generator_model

