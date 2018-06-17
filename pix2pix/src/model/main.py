import argparse
import datetime
import os
import sys

import keras.backend as K


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
    parser.add_argument('--n_run_of_gen_for_1_run_of_disc', default=1, type=int, help="After training disc on 1 batch, how many batches should gen train on")
    parser.add_argument('--save_weights_every_n_epochs', default=10, type=int, help="Epoch at which weights will be saved")
    parser.add_argument('--visualize_images_every_n_epochs', default=10, type=int, help="Epoch at which images and graphs will be visualized")
    parser.add_argument('--use_mbd', action="store_true", help="Whether to use minibatch discrimination")
    parser.add_argument('--use_label_smoothing', action="store_true", help="Whether to smooth the positive labels when training D")
    parser.add_argument('--label_flipping_prob', default=0, type=float, help="Probability (0 to 1.) to flip the labels when training D")
    parser.add_argument('--use_l1_weighted_loss', action="store_true", help="Whether to use l1 loss additionally weighted by mouth position (def: False)")
    parser.add_argument('--prev_model', default=None, type=str, help="model_name of previous model to load latest weights of")
    parser.add_argument('--discriminator_optimizer', default='sgd', type=str, help="discriminator_optimizer: sgd or (default) adam")
    parser.add_argument('--MAX_FRAMES_PER_GIF', default=100, type=int, help="Max number of frames to be saved in each gif")
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
    parser.add_argument('--img_dim', type=int, nargs=3, default=[256, 256, 3], help="Patch size for D")
    parser.add_argument('--patch_size', type=int, nargs=2, default=[64, 64], help="Patch size for D")
    parser.add_argument('--backend', type=str, default="tensorflow", help="theano or tensorflow")
    parser.add_argument('--generator_type', type=str, default="upsampling", help="upsampling or deconv")
    parser.add_argument('--dset', type=str, default="facades", help="dir with 'train' and 'val' dirs with faces_combined images, OR .h5 data file name in the data/processed dir")
    parser.add_argument('--use_identity_image', action="store_true", help="Use identity image")
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size')
    parser.add_argument('--n_batch_per_epoch', default=100, type=int, help="Number of training epochs")
    parser.add_argument('--nb_epoch', default=400, type=int, help="Number of batches per epoch")
    parser.add_argument('--dont_augment_data', action="store_true", help="Use data augmentation while training")
    parser.add_argument('--save_weights_every_n_epochs', default=10, type=int, help="Epoch at which weights will be saved")
    parser.add_argument('--visualize_images_every_n_epochs', default=10, type=int, help="Epoch at which images and graphs will be visualized")
    parser.add_argument('--save_only_last_n_weights', default=20, type=int, help="Save only the last n .h5 files (of, gen, disc and DCGAN each)")
    parser.add_argument('--use_mbd', action="store_true", help="Whether to use minibatch discrimination")
    parser.add_argument('--use_label_smoothing', action="store_true", help="Whether to smooth the positive labels when training D")
    parser.add_argument('--label_flipping_prob', default=0, type=float, help="Probability (0 to 1.) to flip the labels when training D")
    parser.add_argument('--use_l1_weighted_loss', action="store_true", help="Whether to use l1 loss additionally weighted by mouth position (default: False)")
    parser.add_argument('--train_only_generator', action="store_true", help="Whether to train only generator, instead of DCGAN (default: False)")
    parser.add_argument('--prev_model', default=None, type=str, help="model_name of previous model to load latest weights of")
    parser.add_argument('--change_model_name_to_prev_model', action="store_true", help="To change the model name to previous model")
    parser.add_argument('--discriminator_optimizer', default='sgd', type=str, help="discriminator_optimizer: sgd or (default) adam")
    parser.add_argument('--n_run_of_gen_for_1_run_of_disc', default=1, type=int, help="After training disc on 1 batch, how many batches should gen train on")
    parser.add_argument('--load_all_data_at_once', action="store_true", help="To load full data all at once from a .h5 file")
    parser.add_argument('--MAX_FRAMES_PER_GIF', default=100, type=int, help="Max number of frames to be saved in each gif")

    args = parser.parse_args()
    print(args)

    # args = parse_my_args()
    # EXAMPLE:
    # python3 -i main.py --dset ../../data/andrew_ng --batch_size 8 --n_batch_per_epoch 4 --nb_epoch 20000 --dont_augment_data --save_weights_every_n_epochs 10 --visualize_images_every_n_epochs 10 --use_mbd --use_label_smoothing --label_flipping_prob 0.1 --use_l1_weighted_loss --prev_model 2018_07_31_harry_potter

    # python3 -i main.py --dset ../../data/andrew_ng_new --batch_size 8 --n_batch_per_epoch 4 --nb_epoch 20000 --dont_augment_data --save_weights_every_n_epochs 10 --visualize_images_every_n_epochs 10 --train_only_generator

    # Set the backend by modifying the env variable
    if args.backend == "theano":
        os.environ["KERAS_BACKEND"] = "theano"
    elif args.backend == "tensorflow":
        os.environ["KERAS_BACKEND"] = "tensorflow"

    # Import the backend

    # manually set dim ordering otherwise it is not changed
    if args.backend == "theano":
        image_data_format = "channels_first"
        K.set_image_data_format(image_data_format)
    elif args.backend == "tensorflow":
        image_data_format = "channels_last"
        K.set_image_data_format(image_data_format)

    model_name_prefix = "{:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

    if args.load_all_data_at_once:
        model_name = model_name_prefix + '_' + args.dset
    else:
        model_name = model_name_prefix + '_' + os.path.basename(args.dset)

    print("\n\nMODEL NAME: ", model_name, '\n\n')

    # Set default params
    d_params = {"command": " ".join(["python"] + sys.argv),
                "img_dim": args.img_dim,
                "patch_size": args.patch_size,
                "image_data_format": image_data_format,
                "generator_type": args.generator_type,
                "dset": args.dset,
                "use_identity_image": args.use_identity_image,
                "batch_size": args.batch_size,
                "n_batch_per_epoch": args.n_batch_per_epoch,
                "nb_epoch": args.nb_epoch,
                "augment_data": (not args.dont_augment_data),
                "model_name": model_name,
                "save_weights_every_n_epochs": args.save_weights_every_n_epochs,
                "visualize_images_every_n_epochs": args.visualize_images_every_n_epochs,
                "save_only_last_n_weights": args.save_only_last_n_weights,
                "use_mbd": args.use_mbd,
                "use_label_smoothing": args.use_label_smoothing,
                "label_flipping_prob": args.label_flipping_prob,
                "use_l1_weighted_loss": args.use_l1_weighted_loss,
                "train_only_generator": args.train_only_generator,
                "prev_model": args.prev_model,
                "change_model_name_to_prev_model": args.change_model_name_to_prev_model,
                "discriminator_optimizer": args.discriminator_optimizer,
                "n_run_of_gen_for_1_run_of_disc": args.n_run_of_gen_for_1_run_of_disc,
                "load_all_data_at_once" : args.load_all_data_at_once,
                "MAX_FRAMES_PER_GIF": args.MAX_FRAMES_PER_GIF
                }

    # Launch training
    if args.train_only_generator:
        import train_only_generator
        generator_model = train_only_generator.train(**d_params)
    else:
        import train
        generator_model = train.train(**d_params)
    
    # print(d_params)
    
