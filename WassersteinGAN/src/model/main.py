import os
import argparse
import datetime


def launch_training(**kwargs):

    # Launch training
    if kwargs["dset"] == "toy":
        train_WGAN.train_toy(**kwargs)
    else:
        train_WGAN.train(**kwargs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--backend', type=str, default="tensorflow", help="theano or tensorflow")
    parser.add_argument('--generator', type=str, default="upsampling", help="upsampling or deconv or subpixel")
    parser.add_argument('--discriminator', type=str, default="discriminator", help="discriminator discriminator_resnet")
    parser.add_argument('--dset', type=str, default="mnist", help="mnist or celebA or cifar10 or toy")
    parser.add_argument('--img_dim', default=64, type=int, help="Image width == height")
    parser.add_argument('--nb_epoch', default=2000, type=int, help="Number of epochs")
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('--n_batch_per_epoch', default=25, type=int, help="Number of batch per epochs")
    parser.add_argument('--noise_dim', default=100, type=int, help="noise sampler dimension")
    parser.add_argument('--noise_scale', default=0.5, type=float, help="noise sampler variance")
    parser.add_argument('--disc_iterations', default=5, type=int, help="Number of discriminator iterations")
    parser.add_argument('--clamp_lower', default=-0.01, type=float, help="Clamp weights below this value")
    parser.add_argument('--clamp_upper', default=0.01, type=float, help="Clamp weights above this value")
    parser.add_argument('--opt_D', type=str, default="RMSprop", help="Optimizer for the discriminator")
    parser.add_argument('--opt_G', type=str, default="RMSprop", help="Optimizer for the generator")
    parser.add_argument('--lr_D', type=float, default=5E-5, help="learning rate for the discriminator")
    parser.add_argument('--lr_G', type=float, default=5E-5, help="learning rate for the generator")
    parser.add_argument('--use_mbd', action="store_true", help="use mini batch disc")
    parser.add_argument('--save_weights_every_n_epochs', default=10, type=int, help="Choose freq of saving weights")
    parser.add_argument('--save_only_last_n_weights', default=5, type=int, help="Choose number of weights to keep")
    parser.add_argument('--visualize_images_every_n_epochs', default=10, type=int, help="Choose freq of epochs to plot")

    args = parser.parse_args()

    # assert args.dset in ["mnist", "celebA", "cifar10", "toy"]
    assert args.opt_G in ["RMSprop", "SGD", "Adam", "AdamWithWeightnorm"], "Unsupported optimizer"
    assert args.opt_D in ["RMSprop", "SGD", "Adam", "AdamWithWeightnorm"], "Unsupported optimizer"

    # Set the backend by modifying the env variable
    if args.backend == "theano":
        os.environ["KERAS_BACKEND"] = "theano"
    elif args.backend == "tensorflow":
        os.environ["KERAS_BACKEND"] = "tensorflow"

    # Import the backend
    import keras.backend as K

    # manually set dim ordering otherwise it is not changed
    if args.backend == "theano":
        image_data_format = "channels_first"
        K.set_image_data_format(image_data_format)
    elif args.backend == "tensorflow":
        image_data_format = "channels_last"
        K.set_image_data_format(image_data_format)

    import train_WGAN

    model_name = "{0:%Y%m%d_%H%M%S}_WGAN_{1}".format(datetime.datetime.now(), os.path.basename(args.dset.rstrip('/')))
    print("\n\nMODEL NAME:", model_name, '\n\n')

    # Set default params
    d_params = {"generator": args.generator,
                "discriminator": args.discriminator,
                "dset": args.dset,
                "img_dim": args.img_dim,
                "nb_epoch": args.nb_epoch,
                "batch_size": args.batch_size,
                "n_batch_per_epoch": args.n_batch_per_epoch,
                "noise_dim": args.noise_dim,
                "noise_scale": args.noise_scale,
                "disc_iterations": args.disc_iterations,
                "clamp_lower": args.clamp_lower,
                "clamp_upper": args.clamp_upper,
                "lr_D": args.lr_D,
                "lr_G": args.lr_G,
                "opt_D": args.opt_D,
                "opt_G": args.opt_G,
                "use_mbd": args.use_mbd,
                "image_data_format": image_data_format,
                "save_weights_every_n_epochs": args.save_weights_every_n_epochs,
                "save_only_last_n_weights": args.save_only_last_n_weights,
                "visualize_images_every_n_epochs": args.visualize_images_every_n_epochs,
                "model_name": model_name
                }

    # Launch training
    launch_training(**d_params)
