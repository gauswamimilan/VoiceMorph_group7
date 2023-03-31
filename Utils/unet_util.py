import tensorflow as tf


def conv_block(
    x,
    kernels,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="same",
    is_bn=True,
    is_relu=True,
    n=2,
):
    """Custom function for conv2d:
    Apply  3*3 convolutions with BN and relu.
    """
    for i in range(1, n + 1):
        x = tf.keras.layers.Conv2D(
            filters=kernels,
            kernel_size=kernel_size,
            padding=padding,
            strides=strides,
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            kernel_initializer=tf.keras.initializers.he_normal(seed=5),
        )(x)
        if is_bn:
            x = tf.keras.layers.BatchNormalization()(x)
        if is_relu:
            x = tf.keras.activations.relu(x)

    return x


def dot_product(seg, cls):
    b, h, w, n = tf.keras.backend.int_shape(seg)
    seg = tf.reshape(seg, [-1, h * w, n])
    final = tf.einsum("ijk,ik->ijk", seg, cls)
    final = tf.reshape(final, [-1, h, w, n])
    return final


def unet_generator(
    input_shape,
    output_shape,
    output_channels,
    initial_filter_size=64,
    no_of_layers=5,
    training=False,
    deep_supervision=False,
):
    """
    UNet Generator
        :param input_shape: Input shape of the image ( H, W, C )
        :param output_channels: Number of output channels ( 3 for RGB, 1 for grayscale )
        :param initial_filter_size: Number of filters in the first layer
        :param no_of_layers: Number of layers in the network ( for 5: 4 encoder + 1 bottleneck + 4 decoder )
        :param training: Boolean to indicate training or inference
        :return: UNet model


    """
    filters = [initial_filter_size * (2**i) for i in range(no_of_layers)]
    upsample_channels = no_of_layers * initial_filter_size

    input_layer = tf.keras.layers.Input(shape=input_shape)

    # Encoder
    encoder_layers = []
    x = input_layer
    for i, f in enumerate(filters[:-1]):
        x = conv_block(x, f, n=2)
        encoder_layers.append(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    # Bottleneck
    bottle_neck_layer = conv_block(x, filters[-1], n=2)

    # Decoder
    # they are stored in reverse order
    decoders = []
    for i, f in enumerate(filters[:-1]):
        layer_postion = len(filters) - i - 2
        # print(i, f, layer_postion)

        concat_layers = []

        for j in range(len(filters)):
            if j < layer_postion:
                pool_size = 2 ** (layer_postion - j)
                pool_size = (pool_size, pool_size)

                x = encoder_layers[j]
                x = tf.keras.layers.MaxPool2D(pool_size=pool_size)(x)
                x = conv_block(x, initial_filter_size, n=1)
                concat_layers.append(x)

            if j == layer_postion:
                x = encoder_layers[j]
                x = conv_block(x, initial_filter_size, n=1)
                concat_layers.append(x)

            if j > layer_postion:
                upsampling_size = 2 ** (j - layer_postion)
                upsampling_size = (upsampling_size, upsampling_size)

                if j == len(filters) - 1:
                    x = bottle_neck_layer
                else:

                    x = decoders[::-1][j - layer_postion - 1]

                x = tf.keras.layers.UpSampling2D(
                    size=upsampling_size, interpolation="bilinear"
                )(x)
                x = conv_block(x, initial_filter_size, n=1)
                concat_layers.append(x)

        x = tf.keras.layers.Concatenate()(concat_layers)
        x = conv_block(x, upsample_channels, n=1)
        decoders.append(x)

    # Output
    last_layer = decoders[-1]
    last_layer = conv_block(
        last_layer, output_channels, n=1, is_bn=False, is_relu=False
    )

    last_layer = tf.keras.layers.Resizing(
        output_shape[0], output_shape[1], interpolation="bilinear"
    )(last_layer)

    # last_layer = tf.keras.activations.sigmoid(last_layer)

    if deep_supervision:
        if training:
            for i, f in enumerate(decoders[:-1]):
                upsampling_size = 2 ** (len(decoders) - i - 1)
                upsampling_size = (upsampling_size, upsampling_size)

                # print(i, upsampling_size)

                x = decoders[i]
                x = conv_block(x, output_channels, n=1, is_bn=False, is_relu=False)
                x = tf.keras.layers.UpSampling2D(
                    size=upsampling_size, interpolation="bilinear"
                )(x)

                x = tf.keras.layers.Resizing(
                    output_shape[0], output_shape[1], interpolation="bilinear"
                )(x)

                # x = tf.keras.activations.sigmoid(x)
                decoders[i] = x

            decoders[-1] = last_layer

            # bottle neck layer deep supervision
            upsampling_size = 2 ** (len(decoders))
            upsampling_size = (upsampling_size, upsampling_size)
            bottle_neck_layer = conv_block(
                bottle_neck_layer, output_channels, n=1, is_bn=False, is_relu=False
            )
            bottle_neck_layer = tf.keras.layers.UpSampling2D(
                size=upsampling_size, interpolation="bilinear"
            )(bottle_neck_layer)

            bottle_neck_layer = tf.keras.layers.Resizing(
                output_shape[0], output_shape[1], interpolation="bilinear"
            )(bottle_neck_layer)

            # bottle_neck_layer = tf.keras.activations.sigmoid(bottle_neck_layer)

            # all decoders + bottle neck layer
            outputs = decoders + [bottle_neck_layer]

            return tf.keras.Model(
                inputs=input_layer, outputs=outputs, name="Unet3_deep_supervision"
            )
        else:
            return tf.keras.Model(
                inputs=input_layer,
                outputs=[
                    last_layer,
                ],
                name="Unet3_deep_supervision",
            )
    else:
        return tf.keras.Model(
            inputs=input_layer,
            outputs=[
                last_layer,
            ],
            name="Unet3",
        )
