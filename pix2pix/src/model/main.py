import argparse
import os
import datetime


def launch_training(**kwargs):
    # Launch training
    train.train(**kwargs)


def parse_my_args(patch_size=[64, 64], backend='tensorflow', generator_type='upsampling',
                  dset='Mahesh_Babu_black_mouth_polygons', batch_size=2, n_batch_per_epoch=2, nb_epoch=2000, save_weights_every_n_epochs=10):
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('patch_size', type=int, nargs=2, action="store", help="Patch size for D")
    parser.add_argument('--backend', type=str, default="tensorflow", help="theano or tensorflow")
    parser.add_argument('--generator_type', type=str, default="upsampling", help="upsampling or deconv")
    parser.add_argument('--dset', type=str, default="facades", help="facades")
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size')
    parser.add_argument('--n_batch_per_epoch', default=100, type=int, help="Number of training epochs")
    parser.add_argument('--nb_epoch', default=400, type=int, help="Number of batches per epoch")
    parser.add_argument('--save_weights_every_n_epochs', default=10, type=int, help="Epoch at which weights will be saved")
    parser.add_argument('--visualize_images_every_n_epochs', default=1, type=int, help="Epoch at which images and graphs will be visualized")
    parser.add_argument('--use_mbd', action="store_true", help="Whether to use minibatch discrimination")
    parser.add_argument('--use_label_smoothing', action="store_true", help="Whether to smooth the positive labels when training D")
    parser.add_argument('--label_flipping', default=0, type=float, help="Probability (0 to 1.) to flip the labels when training D")
    parser.add_argument('--prev_model', default=None, type=str, help="model_name of previous model to load latest weights of")
    # parser.add_argument('--cfg', action="store", dest="cfg",
    #                     help='cfg model file (/path/to/model_config.yaml)', type=str)
    # parser.add_argument('--wts', action="store", dest="weights",
    #                     help='weights model file (/path/to/model_weights.pkl)', type=str)
    # parser.add_argument('--output-dir', action="store", dest="output_dir",
    #                     help='directory for visualization pdfs (default: /tmp/infer_simple)', type=str)
    # parser.add_argument('--image-ext', action="store", dest="image_ext",
    #                     help='image file name extension (default: jpg)', type=str)
    # parser.add_argument('im_or_folder', help='image or folder of images', default=None)
    # output_dir = "/home/play/data/detectron/voletiv/driveai/" + ".".join(yaml_file.split('/')[-1].split('.')[:-1])
    return parser.parse_args([str(patch_size[0]), str(patch_size[1]),
                              '--backend', backend,
                              '--generator_type', generator_type,
                              '--dset', dset,
                              '--batch_size', str(batch_size),
                              '--n_batch_per_epoch', str(n_batch_per_epoch),
                              '--nb_epoch', str(nb_epoch),
                              '--save_weights_every_n_epochs', str(save_weights_every_n_epochs),
                              '--prev_model', '1520525495_Mahesh_Babu_black_mouth_polygons'
                              ])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('patch_size', type=int, nargs=2, action="store", help="Patch size for D")
    parser.add_argument('--backend', type=str, default="tensorflow", help="theano or tensorflow")
    parser.add_argument('--generator_type', type=str, default="upsampling", help="upsampling or deconv")
    parser.add_argument('--dset', type=str, default="facades", help="facades")
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size')
    parser.add_argument('--n_batch_per_epoch', default=100, type=int, help="Number of training epochs")
    parser.add_argument('--nb_epoch', default=400, type=int, help="Number of batches per epoch")
    parser.add_argument('--save_weights_every_n_epochs', default=10, type=int, help="Epoch at which weights will be saved")
    parser.add_argument('--visualize_images_every_n_epochs', default=1, type=int, help="Epoch at which images and graphs will be visualized")
    parser.add_argument('--use_mbd', action="store_true", help="Whether to use minibatch discrimination")
    parser.add_argument('--use_label_smoothing', action="store_true", help="Whether to smooth the positive labels when training D")
    parser.add_argument('--label_flipping', default=0, type=float, help="Probability (0 to 1.) to flip the labels when training D")
    parser.add_argument('--prev_model', default=None, type=str, help="model_name of previous model to load latest weights of")

    args = parser.parse_args()
    print(args)

    # args = parse_my_args()

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

    model_name = "{:%Y%m%d_%H%M%S}".format(datetime.datetime.now()) + '_' + args.dset
    print("\n\nMODEL NAME: ", model_name, '\n\n')

    # Set default params
    d_params = {"dset": args.dset,
                "generator_type": args.generator_type,
                "batch_size": args.batch_size,
                "n_batch_per_epoch": args.n_batch_per_epoch,
                "nb_epoch": args.nb_epoch,
                "model_name": model_name,
                "save_weights_every_n_epochs": args.save_weights_every_n_epochs,
                "visualize_images_every_n_epochs": visualize_images_every_n_epochs,
                "image_data_format": image_data_format,
                "use_label_smoothing": args.use_label_smoothing,
                "label_flipping": args.label_flipping,
                "patch_size": args.patch_size,
                "use_mbd": args.use_mbd,
                "prev_model": args.prev_model
                }

    # Launch training
    launch_training(**d_params)
