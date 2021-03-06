from keras.models import Model
from keras.layers.core import Flatten, Dense, Dropout, Activation, Lambda, Reshape
from keras.layers.convolutional import Conv2D, Deconv2D, ZeroPadding2D, UpSampling2D
# from keras.layers import Input, merge
from keras.layers import Input, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
import keras.backend as K


def generator_upsampling(cat_dim, cont_dim, noise_dim, img_dim, model_name="generator_upsampling", dset="mnist"):
    """
    Generator model of the DCGAN

    args : img_dim (tuple of int) num_chan, height, width
           pretr_weights_file (str) file holding pre trained weights

    returns : model (keras NN) the Neural Net model
    """

    s = img_dim[1]
    f = 128

    if dset == "mnist":
        start_dim = int(s / 4)
        nb_upconv = 2
    elif dset == "celebA":
        start_dim = int(s / 16)
        nb_upconv = 4
    else:
        o = s
        nb_upconv = 0
        while o > 7:
            o = o/2
            nb_upconv += 1
        start_dim = int(o)

    if K.image_dim_ordering() == "th":
        bn_axis = 1
        reshape_shape = (f, start_dim, start_dim)
        output_channels = img_dim[0]
    else:
        reshape_shape = (start_dim, start_dim, f)
        bn_axis = -1
        output_channels = img_dim[-1]

    cat_input = Input(shape=cat_dim, name="cat_input")
    cont_input = Input(shape=cont_dim, name="cont_input")
    noise_input = Input(shape=noise_dim, name="noise_input")

    # gen_input = merge([cat_input, cont_input, noise_input], mode="concat")
    gen_input = Concatenate()([noise_input, cont_input, cat_input])

    x = Dense(1024)(gen_input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Dense(f * start_dim * start_dim)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Reshape(reshape_shape)(x)

    # Upscaling blocks
    for i in range(nb_upconv):
        x = UpSampling2D(size=(2, 2))(x)
        nb_filters = int(f / (2 ** (i + 1)))
        x = Conv2D(nb_filters, (3, 3), padding="same")(x)
        x = BatchNormalization(axis=bn_axis)(x)
        x = Activation("relu")(x)
        # x = Conv2D(nb_filters, (3, 3), padding="same")(x)
        # x = BatchNormalization(axis=bn_axis)(x)
        # x = Activation("relu")(x)

    x = Conv2D(output_channels, (3, 3), name="gen_Conv2D_final", padding="same", activation='tanh')(x)

    generator_model = Model(inputs=[cat_input, cont_input, noise_input], outputs=[x], name=model_name)

    return generator_model


def generator_deconv(cat_dim, cont_dim, noise_dim, img_dim, batch_size, model_name="generator_deconv", dset="mnist"):
    """
    Generator model of the DCGAN

    args : nb_classes (int) number of classes
           img_dim (tuple of int) num_chan, height, width
           pretr_weights_file (str) file holding pre trained weights

    returns : model (keras NN) the Neural Net model
    """

    assert K.backend() == "tensorflow", "Deconv not implemented with theano"

    s = img_dim[1]
    f = 128

    if dset == "mnist":
        start_dim = int(s / 4)
        nb_upconv = 2
    else:
        start_dim = int(s / 16)
        nb_upconv = 4

    reshape_shape = (start_dim, start_dim, f)
    bn_axis = -1
    output_channels = img_dim[-1]

    cat_input = Input(shape=cat_dim, name="cat_input")
    cont_input = Input(shape=cont_dim, name="cont_input")
    noise_input = Input(shape=noise_dim, name="noise_input")

    # gen_input = merge([cat_input, cont_input, noise_input], mode="concat")
    gen_input = Concatenate()([noise_input, cont_input, cat_input])

    x = Dense(1024)(gen_input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Dense(f * start_dim * start_dim)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Reshape(reshape_shape)(x)

    # Transposed conv blocks
    for i in range(nb_upconv - 1):
        nb_filters = int(f / (2 ** (i + 1)))
        s = start_dim * (2 ** (i + 1))
        o_shape = (batch_size, s, s, nb_filters)
        x = Deconv2D(nb_filters, (3, 3), output_shape=o_shape, strides=(2, 2), padding="same")(x)
        x = BatchNormalization(mode=2, axis=bn_axis)(x)
        x = Activation("relu")(x)

    # Last block
    s = start_dim * (2 ** (nb_upconv))
    o_shape = (batch_size, s, s, output_channels)
    x = Deconv2D(output_channels, (3, 3), output_shape=o_shape, strides=(2, 2), padding="same")(x)
    x = Activation("tanh")(x)

    generator_model = Model(inputs=[cat_input, cont_input, noise_input], outputs=[x], name=model_name)

    return generator_model


def DCGAN_discriminator(cat_dim, cont_dim, img_dim, model_name="DCGAN_discriminator", dset="mnist", use_mbd=False):
    """
    Discriminator model of the DCGAN

    args : img_dim (tuple of int) num_chan, height, width
           pretr_weights_file (str) file holding pre trained weights

    returns : model (keras NN) the Neural Net model
    """

    if K.image_dim_ordering() == "th":
        bn_axis = 1
    else:
        bn_axis = -1

    disc_input = Input(shape=img_dim, name="discriminator_input")

    if dset == "mnist":
        list_f = [128]

    else:
        list_f = [64, 128, 256]

    # First conv
    x = Conv2D(64, (3, 3), strides=(2, 2), name="disc_Conv2D_1", padding="same")(disc_input)
    x = LeakyReLU(0.2)(x)

    # Next convs
    for i, f in enumerate(list_f):
        name = "disc_Conv2D_%s" % (i + 2)
        x = Conv2D(f, (3, 3), strides=(2, 2), name=name, padding="same")(x)
        x = BatchNormalization(axis=bn_axis)(x)
        x = LeakyReLU(0.2)(x)

    x = Flatten()(x)
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    def linmax(x):
        return K.maximum(x, -16)

    def linmax_shape(input_shape):
        return input_shape

    # More processing for auxiliary Q
    x_Q = Dense(128)(x)
    x_Q = BatchNormalization()(x_Q)
    x_Q = LeakyReLU(0.2)(x_Q)
    x_Q_Y = Dense(cat_dim[0], activation='softmax', name="Q_cat_out")(x_Q)
    x_Q_C_mean = Dense(cont_dim[0], activation='linear', name="dense_Q_cont_mean")(x_Q)
    x_Q_C_logstd = Dense(cont_dim[0], name="dense_Q_cont_logstd")(x_Q)
    x_Q_C_logstd = Lambda(linmax, output_shape=linmax_shape)(x_Q_C_logstd)
    # Reshape Q to nbatch, 1, cont_dim[0]
    x_Q_C_mean = Reshape((1, cont_dim[0]))(x_Q_C_mean)
    x_Q_C_logstd = Reshape((1, cont_dim[0]))(x_Q_C_logstd)
    # x_Q_C = merge([x_Q_C_mean, x_Q_C_logstd], mode="concat", name="Q_cont_out", concat_axis=1)
    x_Q_C = Concatenate(name="Q_cont_out", axis=1)([x_Q_C_mean, x_Q_C_logstd])

    def minb_disc(z):
        diffs = K.expand_dims(z, 3) - K.expand_dims(K.permute_dimensions(z, [1, 2, 0]), 0)
        abs_diffs = K.sum(K.abs(diffs), 2)
        z = K.sum(K.exp(-abs_diffs), 2)

        return z

    def lambda_output(input_shape):
        return input_shape[:2]

    num_kernels = 300
    dim_per_kernel = 5

    M = Dense(num_kernels * dim_per_kernel, use_bias=False, activation=None)
    MBD = Lambda(minb_disc, output_shape=lambda_output)

    if use_mbd:
        x_mbd = M(x)
        x_mbd = Reshape((num_kernels, dim_per_kernel))(x_mbd)
        x_mbd = MBD(x_mbd)
        x = merge([x, x_mbd], mode='concat')
        x = Concatenate()([x, x_mbd])

    # Create discriminator model
    x_disc = Dense(2, activation='softmax', name="disc_out")(x)
    discriminator_model = Model(inputs=[disc_input], outputs=[x_disc, x_Q_Y, x_Q_C], name=model_name)

    return discriminator_model


def DCGAN(generator, discriminator_model, cat_dim, cont_dim, noise_dim):

    cat_input = Input(shape=cat_dim, name="cat_input")
    cont_input = Input(shape=cont_dim, name="cont_input")
    noise_input = Input(shape=noise_dim, name="noise_input")

    generated_image = generator([cat_input, cont_input, noise_input])
    x_disc, x_Q_Y, x_Q_C = discriminator_model(generated_image)

    DCGAN = Model(inputs=[cat_input, cont_input, noise_input],
                  outputs=[x_disc, x_Q_Y, x_Q_C],
                  name="DCGAN")

    return DCGAN


def load(model_name, cat_dim, cont_dim, noise_dim, img_dim, batch_size, dset="mnist", use_mbd=False):

    if model_name == "generator_upsampling":
        model = generator_upsampling(cat_dim, cont_dim, noise_dim, img_dim, model_name=model_name, dset=dset)
        model.summary()
        from keras.utils import plot_model
        plot_model(model, to_file='../../figures/%s.png' % model_name, show_shapes=True, show_layer_names=True)
        return model
    if model_name == "generator_deconv":
        model = generator_deconv(cat_dim, cont_dim, noise_dim, img_dim,
                                 batch_size, model_name=model_name, dset=dset)
        model.summary()
        from keras.utils import plot_model
        plot_model(model, to_file='../../figures/%s.png' % model_name, show_shapes=True, show_layer_names=True)
        return model
    if model_name == "DCGAN_discriminator":
        model = DCGAN_discriminator(cat_dim, cont_dim, img_dim,
                                    model_name=model_name, dset=dset, use_mbd=use_mbd)
        model.summary()
        from keras.utils import plot_model
        plot_model(model, to_file='../../figures/%s.png' % model_name, show_shapes=True, show_layer_names=True)
        return model


if __name__ == '__main__':

    m = generator_deconv((10,), (2,), (64,), (28, 28, 1), 2, 1, model_name="generator_deconv", dset="mnist")
    m.summary()
