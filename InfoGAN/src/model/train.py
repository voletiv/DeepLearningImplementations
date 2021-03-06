import datetime
import os
import sys
import time
import models as models

from keras.utils import generic_utils
from keras.optimizers import Adam, SGD
import keras.backend as K

# Utils
sys.path.append("../utils")
import general_utils
import data_utils


def gaussian_loss(y_true, y_pred):

    Q_C_mean = y_pred[:, 0, :]
    Q_C_logstd = y_pred[:, 1, :]

    y_true = y_true[:, 0, :]

    epsilon = (y_true - Q_C_mean) / (K.exp(Q_C_logstd) + K.epsilon())
    loss_Q_C = (Q_C_logstd + 0.5 * K.square(epsilon))
    loss_Q_C = K.mean(loss_Q_C)

    return loss_Q_C


def train(**kwargs):
    """
    Train model

    Load the whole train data in memory for faster operations

    args: **kwargs (dict) keyword arguments that specify the model hyperparameters
    """

    # Roll out the parameters
    batch_size = kwargs["batch_size"]
    n_batch_per_epoch = kwargs["n_batch_per_epoch"]
    nb_epoch = kwargs["nb_epoch"]
    generator = kwargs["generator"]
    model_name = kwargs["model_name"]
    image_data_format = kwargs["image_data_format"]
    celebA_img_dim = kwargs["celebA_img_dim"]
    cont_dim = (kwargs["cont_dim"],)
    cat_dim = (kwargs["cat_dim"],)
    noise_dim = (kwargs["noise_dim"],)
    label_smoothing = kwargs["label_smoothing"]
    label_flipping = kwargs["label_flipping"]
    noise_scale = kwargs["noise_scale"]
    dset = kwargs["dset"]
    use_mbd = kwargs["use_mbd"]
    load_from_dir = kwargs["load_from_dir"]
    target_size = kwargs["target_size"]
    save_weights_every_n_epochs = kwargs["save_weights_every_n_epochs"]
    save_only_last_n_weights = kwargs["save_only_last_n_weights"]
    visualize_images_every_n_epochs = kwargs["visualize_images_every_n_epochs"]
    epoch_size = n_batch_per_epoch * batch_size

    # Setup environment (logging directory etc)
    general_utils.setup_logging(**kwargs)

    # Load and rescale data
    if dset == "celebA":
        X_real_train = data_utils.load_celebA(celebA_img_dim, image_data_format)
    elif dset == "mnist":
        X_real_train, _, _, _ = data_utils.load_mnist(image_data_format)
    else:
        X_batch_gen = data_utils.data_generator_from_dir(dset, target_size, batch_size)
        X_real_train = next(X_batch_gen)
    img_dim = X_real_train.shape[-3:]

    try:

        # Create optimizers
        opt_dcgan = Adam(lr=1E-4, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
        opt_discriminator = Adam(lr=1E-4, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
        # opt_discriminator = SGD(lr=1E-4, momentum=0.9, nesterov=True)

        # Load generator model
        generator_model = models.load("generator_%s" % generator,
                                      cat_dim,
                                      cont_dim,
                                      noise_dim,
                                      img_dim,
                                      batch_size,
                                      dset=dset,
                                      use_mbd=use_mbd)
        # Load discriminator model
        discriminator_model = models.load("DCGAN_discriminator",
                                          cat_dim,
                                          cont_dim,
                                          noise_dim,
                                          img_dim,
                                          batch_size,
                                          dset=dset,
                                          use_mbd=use_mbd)

        generator_model.compile(loss='mse', optimizer=opt_discriminator)
        discriminator_model.trainable = False

        DCGAN_model = models.DCGAN(generator_model,
                                   discriminator_model,
                                   cat_dim,
                                   cont_dim,
                                   noise_dim)

        list_losses = ['binary_crossentropy', 'categorical_crossentropy', gaussian_loss]
        list_weights = [1, 1, 1]
        DCGAN_model.compile(loss=list_losses, loss_weights=list_weights, optimizer=opt_dcgan)

        # Multiple discriminator losses
        discriminator_model.trainable = True
        discriminator_model.compile(loss=list_losses, loss_weights=list_weights, optimizer=opt_discriminator)

        gen_loss = 100
        disc_loss = 100

        if not load_from_dir:
            X_batch_gen = data_utils.gen_batch(X_real_train, batch_size)

        # Start training
        print("Start training")

        disc_total_losses = []
        disc_log_losses = []
        disc_cat_losses = []
        disc_cont_losses = []
        gen_total_losses = []
        gen_log_losses = []
        gen_cat_losses = []
        gen_cont_losses = []

        start = time.time()

        for e in range(nb_epoch):

            print('--------------------------------------------')
            print('[{0:%Y/%m/%d %H:%M:%S}] Epoch {1:d}/{2:d}\n'.format(datetime.datetime.now(), e + 1, nb_epoch))

            # Initialize progbar and batch counter
            progbar = generic_utils.Progbar(epoch_size)
            batch_counter = 1

            disc_total_loss_batch = 0
            disc_log_loss_batch = 0
            disc_cat_loss_batch = 0
            disc_cont_loss_batch = 0
            gen_total_loss_batch = 0
            gen_log_loss_batch = 0
            gen_cat_loss_batch = 0
            gen_cont_loss_batch = 0

            for batch_counter in range(n_batch_per_epoch):

                # Load data
                X_real_batch = next(X_batch_gen)

                # Create a batch to feed the discriminator model
                X_disc, y_disc, y_cat, y_cont = data_utils.get_disc_batch(X_real_batch,
                                                                          generator_model,
                                                                          batch_counter,
                                                                          batch_size,
                                                                          cat_dim,
                                                                          cont_dim,
                                                                          noise_dim,
                                                                          noise_scale=noise_scale,
                                                                          label_smoothing=label_smoothing,
                                                                          label_flipping=label_flipping)

                # Update the discriminator
                disc_loss = discriminator_model.train_on_batch(X_disc, [y_disc, y_cat, y_cont])

                # Create a batch to feed the generator model
                X_gen, y_gen, y_cat, y_cont, y_cont_target = data_utils.get_gen_batch(batch_size,
                                                                                      cat_dim,
                                                                                      cont_dim,
                                                                                      noise_dim,
                                                                                      noise_scale=noise_scale)

                # Freeze the discriminator
                discriminator_model.trainable = False
                gen_loss = DCGAN_model.train_on_batch([y_cat, y_cont, X_gen], [y_gen, y_cat, y_cont_target])
                # Unfreeze the discriminator
                discriminator_model.trainable = True

                progbar.add(batch_size, values=[("D tot", disc_loss[0]),
                                                ("D log", disc_loss[1]),
                                                ("D cat", disc_loss[2]),
                                                ("D cont", disc_loss[3]),
                                                ("G tot", gen_loss[0]),
                                                ("G log", gen_loss[1]),
                                                ("G cat", gen_loss[2]),
                                                ("G cont", gen_loss[3])])

                disc_total_loss_batch += disc_loss[0]
                disc_log_loss_batch += disc_loss[1]
                disc_cat_loss_batch += disc_loss[2]
                disc_cont_loss_batch += disc_loss[3]
                gen_total_loss_batch += gen_loss[0]
                gen_log_loss_batch += gen_loss[1]
                gen_cat_loss_batch += gen_loss[2]
                gen_cont_loss_batch += gen_loss[3]

                # # Save images for visualization
                # if batch_counter % (n_batch_per_epoch / 2) == 0:
                #     data_utils.plot_generated_batch(X_real_batch, generator_model, e,
                #                                     batch_size, cat_dim, cont_dim, noise_dim,
                #                                     image_data_format, model_name)

            disc_total_losses.append(disc_total_loss_batch/n_batch_per_epoch)
            disc_log_losses.append(disc_log_loss_batch/n_batch_per_epoch)
            disc_cat_losses.append(disc_cat_loss_batch/n_batch_per_epoch)
            disc_cont_losses.append(disc_cont_loss_batch/n_batch_per_epoch)
            gen_total_losses.append(gen_total_loss_batch/n_batch_per_epoch)
            gen_log_losses.append(gen_log_loss_batch/n_batch_per_epoch)
            gen_cat_losses.append(gen_cat_loss_batch/n_batch_per_epoch)
            gen_cont_losses.append(gen_cont_loss_batch/n_batch_per_epoch)

            # Save images for visualization
            if (e + 1) % visualize_images_every_n_epochs == 0:
                data_utils.plot_generated_batch(X_real_batch, generator_model, e, batch_size,
                                                cat_dim, cont_dim, noise_dim, image_data_format, model_name)
                data_utils.plot_losses(disc_total_losses, disc_log_losses, disc_cat_losses, disc_cont_losses,
                                       gen_total_losses, gen_log_losses, gen_cat_losses, gen_cont_losses,
                                       model_name)

            if (e + 1) % save_weights_every_n_epochs == 0:

                print("Saving weights...")

                # Delete all but the last n weights
                general_utils.purge_weights(save_only_last_n_weights, model_name)

                # Save weights
                gen_weights_path = os.path.join('../../models/%s/gen_weights_epoch%05d.h5' % (model_name, e))
                generator_model.save_weights(gen_weights_path, overwrite=True)

                disc_weights_path = os.path.join('../../models/%s/disc_weights_epoch%05d.h5' % (model_name, e))
                discriminator_model.save_weights(disc_weights_path, overwrite=True)

                DCGAN_weights_path = os.path.join('../../models/%s/DCGAN_weights_epoch%05d.h5' % (model_name, e))
                DCGAN_model.save_weights(DCGAN_weights_path, overwrite=True)

            end = time.time()
            print("")
            print('Epoch %s/%s, Time: %s' % (e + 1, nb_epoch, end - start))
            start = end

    except KeyboardInterrupt:
        pass

    gen_weights_path = '../../models/%s/generator_latest.h5' % (model_name)
    print("Saving", gen_weights_path)
    generator_model.save(gen_weights_path, overwrite=True)
