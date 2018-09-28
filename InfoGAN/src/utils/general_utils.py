import os
import subprocess


def remove_files(files):
    """
    Remove files from disk

    args: files (str or list) remove all files in 'files'
    """

    if isinstance(files, (list, tuple)):
        for f in files:
            if os.path.isfile(os.path.expanduser(f)):
                os.remove(f)
    elif isinstance(files, str):
        if os.path.isfile(os.path.expanduser(files)):
            os.remove(files)


def create_dir(dirs):
    """
    Create directory

    args: dirs (str or list) create all dirs in 'dirs'
    """

    if isinstance(dirs, (list, tuple)):
        for d in dirs:
            if not os.path.exists(os.path.expanduser(d)):
                os.makedirs(d)
    elif isinstance(dirs, str):
        if not os.path.exists(os.path.expanduser(dirs)):
            os.makedirs(dirs)


def setup_logging(**kwargs):

    model_name = kwargs["model_name"]

    # Output path where we store experiment log and weights
    model_dir = os.path.join("../../models", model_name)
    fig_dir = os.path.join("../../figures", model_name)

    # Create if it does not exist
    print("Creating", model_dir, "and", fig_dir)
    create_dir([model_dir, fig_dir])

    # Copy main.py, train.py and model.py
    subprocess.call(['cp', 'main.py', model_dir])
    subprocess.call(['cp', 'models.py', model_dir])
    subprocess.call(['cp', 'train.py', model_dir])

    # Write all config params
    print("Writing config params in", os.path.join(model_dir, 'config.txt'))
    with open(os.path.join(model_dir, 'config.txt'), 'w') as f:
        for i in kwargs:
            f.write(str(i) + ' ' + str(kwargs[i]) + '\n')

    print("Writing config params in", os.path.join(fig_dir, 'config.txt'))
    with open(os.path.join(fig_dir, 'config.txt'), 'w') as f:
        for i in kwargs:
            f.write(str(i) + ' ' + str(kwargs[i]) + '\n')


def purge_weights(n, model_name):
    gen_weight_files = sorted(glob.glob('../../models/%s/gen_weights*' % model_name))
    for gen_weight_file in gen_weight_files[:-n]:
        os.remove(os.path.realpath(gen_weight_file))

    disc_weight_files = sorted(glob.glob('../../models/%s/disc_weights*' % model_name))
    for disc_weight_file in disc_weight_files[:-n]:
        os.remove(os.path.realpath(disc_weight_file))

    DCGAN_weight_files = sorted(glob.glob('../../models/%s/DCGAN_weights*' % model_name))
    for DCGAN_weight_file in DCGAN_weight_files[:-n]:
        os.remove(os.path.realpath(DCGAN_weight_file))
