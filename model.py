#%%
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import Conv3DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Lambda

class GAN:
    def __init__(self, embeddings, latent_dim):
        input_shape = (51, 61, 23, 1)
        input_a = Input(shape=input_shape)
        input_b = Input(shape=input_shape)
        in_label_a = Input(shape=(1,))
        in_label_b = Input(shape=(1,))

        d_model = self.define_discriminator(embeddings)
        g_model = self.define_generator(embeddings, latent_dim)

        dis_d1 = d_model([input_a, in_label_a])
        dis_d2 = d_model([input_b, in_label_b])

        def difference(inputs):
            x1, x2 = inputs
            return (x1 - x2)

        dis_dif_layer = Lambda(difference)([dis_d1, dis_d2])
        dis_out_layer = Dense(1, activation='sigmoid')(dis_dif_layer)

        self.dis_model = Model([input_a, input_b, in_label_a, in_label_b], outputs=dis_out_layer, name="dis_model")

        input_c = Input(shape=input_shape)
        in_label_c = Input(shape=(1,))
        gen_z_input, gen_label_input = g_model.input
        d_model.trainable = False

        gen_d1 = d_model([input_c, in_label_c])
        gen_d2 = d_model([g_model.output, gen_label_input])

        gen_dif_layer = Lambda(difference)([gen_d1, gen_d2])
        gen_out_layer = Dense(1, activation='sigmoid')(gen_dif_layer)

        self.gen_model = Model([input_c, gen_z_input, in_label_c, gen_label_input], outputs=[gen_out_layer, g_model.output], name="gan_model")

    def compile(self, optimizer, losses, loss_weights):
        self.dis_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.gen_model.compile(loss=losses, loss_weights=loss_weights, optimizer=optimizer, metrics=['accuracy'])

    def define_discriminator(self, embeddings, in_shape=(51, 61, 23, 1), n_classes=60):
        # label input
        in_label = Input(shape=(1,))
        # embedding for categorical input
        embedding_layer = Embedding(input_dim=n_classes, output_dim=25, input_length=1, weights=[embeddings], trainable=False)
        li = embedding_layer(in_label)
        # scale up to image dimensions with linear activation
        n_nodes = in_shape[0] * in_shape[1] * in_shape[2]
        li = Dense(n_nodes)(li)
        # reshape to additional channel
        li = Reshape((in_shape[0], in_shape[1], in_shape[2], 1))(li)
        # image input
        in_image = Input(shape=in_shape)
        # concat label as a channel
        merge = Concatenate()([in_image, li])
        # downsample
        fe = Conv3D(32, (3,3,3), strides=(2,2,2), padding='same')(merge)
        fe = LeakyReLU(alpha=0.2)(fe)
        # downsample
        fe = Conv3D(64, (3,3,3), strides=(2,2,2), padding='same')(fe)
        fe = LeakyReLU(alpha=0.2)(fe)
        # downsample
        fe = Conv3D(128, (3,3,3), strides=(2,2,2), padding='same')(fe)
        fe = LeakyReLU(alpha=0.2)(fe)
        # flatten feature maps
        fe = Flatten()(fe)
        # dropout
        fe = Dropout(0.4)(fe)
        # output
        out_layer = Dense(2048, activation=None)(fe)
        # define model
        model = Model([in_image, in_label], out_layer, name='discriminator')
        # compile model
        # opt = Adam(lr=0.0002, beta_1=0.5)
        # model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    # define the standalone generator model
    def define_generator(self, embeddings, latent_dim, n_classes=60):
        # label input
        in_label = Input(shape=(1,))
        # embedding for categorical input
        embedding_layer = Embedding(input_dim=n_classes, output_dim=25, input_length=1, weights=[embeddings], trainable=False)
        li = embedding_layer(in_label)
        # linear multiplication
        n_nodes = 7 * 7 * 7
        li = Dense(n_nodes)(li)
        # reshape to additional channel
        li = Reshape((7, 7, 7, 1))(li)
        # image generator input
        in_lat = Input(shape=(latent_dim,))
        # foundation for 7x7 image
        n_nodes = 128 * 7 * 7 * 7
        gen = Dense(n_nodes)(in_lat)
        gen = LeakyReLU(alpha=0.2)(gen)
        gen = Reshape((7, 7, 7, 128))(gen)
        # merge image gen and label input
        merge = Concatenate()([gen, li])
        # upsample to 14x21
        gen = Conv3DTranspose(128, (3,3,3), strides=(2,3,1), padding='same')(merge)
        gen = LeakyReLU(alpha=0.2)(gen)
        # # upsample to 28x21
        gen = Conv3DTranspose(128, (3,3,3), strides=(2,1,2), padding='same')(gen)
        gen = LeakyReLU(alpha=0.2)(gen)
        # # upsample to 56x63
        gen = Conv3DTranspose(128, (3,3,3), strides=(2,3,2), padding='same')(gen)
        gen = LeakyReLU(alpha=0.2)(gen)
        # output - 51x51x23
        out_layer = Conv3D(1, (6,3,6), strides=(1,1,1), activation='tanh', padding='valid')(gen)
        # define model
        model = Model([in_lat, in_label], out_layer, name='generator')
        # print(model.summary())
        return model
