import os
import argparse
import datetime


def launch_training(**kwargs):

    # Launch training
    train.train(**kwargs)


def launch_eval(**kwargs):

    # Launch training
    eval.eval(**kwargs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('mode', type=str, help="train or eval")
    parser.add_argument('--backend', type=str, default="tensorflow", help="theano or tensorflow")
    parser.add_argument('--generator', type=str, default="upsampling", help="upsampling or deconv")
    parser.add_argument('--dset', type=str, default="mnist", help="mnist or celebA, or path to images of dataset")
    parser.add_argument('--dont_load_from_dir', action="store_true", help="Whether to load images from dir for every batch")
    parser.add_argument('--target_size', default=256, type=int, help="Target_size to resize every image")
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('--n_batch_per_epoch', default=16, type=int, help="Number of training epochs")
    parser.add_argument('--nb_epoch', default=400, type=int, help="Number of batches per epoch")
    parser.add_argument('--eval_epoch', default=10, type=int, help="Epoch at which weights were saved for evaluation")
    parser.add_argument('--do_plot', default=False, type=bool, help="Debugging plot")
    parser.add_argument('--celebA_img_dim', default=64, type=int, help="Image width == height (only specify for CelebA)")
    parser.add_argument('--noise_dim', default=64, type=int, help="noise dimension")
    parser.add_argument('--cont_dim', default=4, type=int, help="Latent continuous dimensions")
    parser.add_argument('--cat_dim', default=8, type=int, help="Latent categorical dimension")
    parser.add_argument('--noise_scale', default=0.5, type=float,
                        help="variance of the normal from which we sample the noise")
    parser.add_argument('--label_smoothing', action="store_true", help="smooth the positive labels when training D")
    parser.add_argument('--use_mbd', action="store_true", help="use mini batch disc")
    parser.add_argument('--label_flipping', default=0, type=float,
                        help="Probability (0 to 1.) to flip the labels when training D")
    parser.add_argument('--save_weights_every_n_epochs', default=5, type=int, help="Choose freq of saving weights")
    parser.add_argument('--save_only_last_n_weights', default=5, type=int, help="Choose number of weights to keep")

    args = parser.parse_args()

    assert args.mode in ["train", "eval"]
    # assert args.dset in ["mnist", "celebA"]

    if args.dset in ["mnist", "celebA"]:
        load_from_dir = False
    else:
        load_from_dir = not args.dont_load_from_dir

    assert args.target_size in [  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
        13,  14,  16,  18,  20,  22,  24,  26,  28,  32,  36,  40,  44,
        48,  52,  56,  64,  72,  80,  88,  96, 104, 112, 128, 144, 160,
       176, 192, 208, 224, 256, 288, 320, 352, 384, 416, 448, 512, 576,
       640, 704, 768, 832, 896]
    target_size = (args.target_size, args.target_size)

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

    import train
    import eval

    model_name = "InfoGAN_{0:%Y%m%d_%H%M%S}_{1}".format(datetime.datetime.now(), os.path.basename(args.dset))

    # Set default params
    d_params = {"dset": args.dset,
                "generator": args.generator,
                "batch_size": args.batch_size,
                "n_batch_per_epoch": args.n_batch_per_epoch,
                "nb_epoch": args.nb_epoch,
                "model_name": model_name,
                "eval_epoch": args.epoch,
                "do_plot": args.do_plot,
                "image_data_format": image_data_format,
                "celebA_img_dim": args.celebA_img_dim,
                "noise_dim": args.noise_dim,
                "cat_dim": args.cat_dim,
                "cont_dim": args.cont_dim,
                "label_smoothing": args.label_smoothing,
                "label_flipping": args.label_flipping,
                "noise_scale": args.noise_scale,
                "use_mbd": args.use_mbd,
                "load_from_dir": load_from_dir,
                "target_size": target_size,
                "save_weights_every_n_epochs": args.save_weights_every_n_epochs,
                "save_only_last_n_weights": args.save_only_last_n_weights
                }

    if args.mode == "train":
        # Launch training
        launch_training(**d_params)

    if args.mode == "eval":
        # Launch eval
        launch_eval(**d_params)
