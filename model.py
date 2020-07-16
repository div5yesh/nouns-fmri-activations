#%%
import os
from numpy.random import randn, randint
from numpy import expand_dims
from numpy import zeros, ones, ones_like

import tensorflow as tf
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
from tensorflow.keras.optimizers import Adam

class GAN:
    def __init__(self, logger, embeddings, latent_dim):
        self.embeddings = embeddings
        self.latent_dim = latent_dim
        self.logger = logger

    def generate_real_samples(self, dataset, n_samples):
        # split into images and labels
        images, labels = dataset
        # choose random instances
        # ix = choice(classes, n_samples)
        ix = randint(0, images.shape[0], n_samples)
        # select images and labels
        X, labels = images[ix], labels[ix]
        # generate class labels
        # y = ones((n_samples, 1))
        y = randint(7, 12, (n_samples, 1)) / 10
        return [X, labels], y

    # generate points in latent space as input for the generator
    def generate_latent_points(self, dataset, n_samples):
        # generate points in the latent space
        images, labels = dataset
        z_input = tf.random.normal((n_samples, self.latent_dim), mean=0.0, stddev=1.0, dtype=tf.dtypes.float32)
        # generate labels
        ix = randint(0, images.shape[0], n_samples)
        z_labels = labels[ix]
        # z_labels = choice(classes, n_samples)
        return z_input, z_labels, ix
    
    # use the generator to generate n fake examples, with class labels
    def generate_fake_samples(self, generator, dataset, n_samples):
        # generate points in latent space
        z_input, z_labels, _ = self.generate_latent_points(dataset, n_samples)
        # predict outputs
        images = generator.predict([z_input, z_labels])
        # create class labels
        # y = zeros((n_samples, 1))
        y = randint(0, 3, (n_samples, 1)) / 10
        return [images , z_labels], y

    def define_discriminator(self, embeddings, activation, in_shape=(51, 61, 23, 1), n_classes=60):
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
        out_layer = activation(fe)
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

# %%
class RTGAN(GAN):
    def create(self, optimizer, losses, loss_weights):
        activation = Dense(2048, activation=None)
        d_model = self.define_discriminator(self.embeddings, activation)
        g_model = self.define_generator(self.embeddings, self.latent_dim)

        input_shape = (51, 61, 23, 1)
        input_a = Input(shape=input_shape)
        input_b = Input(shape=input_shape)
        in_label_a = Input(shape=(1,))
        in_label_b = Input(shape=(1,))

        dis_d1 = d_model([input_a, in_label_a])
        dis_d2 = d_model([input_b, in_label_b])

        def difference(inputs):
            x1, x2 = inputs
            return (x1 - x2)

        dis_dif_layer = Lambda(difference)([dis_d1, dis_d2])
        dis_out_layer = Dense(1, activation='sigmoid')(dis_dif_layer)

        dis_model = Model([input_a, input_b, in_label_a, in_label_b], outputs=dis_out_layer, name="dis_model")
        dis_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        dis_model.summary()

        input_c = Input(shape=input_shape)
        in_label_c = Input(shape=(1,))
        gen_z_input, gen_label_input = g_model.input

        d_model.trainable = False
        gen_d1 = d_model([input_c, in_label_c])
        gen_d2 = d_model([g_model.output, gen_label_input])

        gen_dif_layer = Lambda(difference)([gen_d1, gen_d2])
        gen_out_layer = Dense(1, activation='sigmoid')(gen_dif_layer)

        gan_model = Model([input_c, gen_z_input, in_label_c, gen_label_input], outputs=[gen_out_layer, g_model.output], name="gan_model")

        gan_model.compile(loss=losses, loss_weights=loss_weights, optimizer=optimizer, metrics=['accuracy'])
        gan_model.summary()

        return g_model, dis_model, gan_model

    def train(self, model_name, g_model, d_model, gan_model, dataset, n_epochs=100, n_batch=2):
        bat_per_epo = int(dataset[0].shape[0] / n_batch)
        # half_batch = int(n_batch / 2)
        # manually enumerate epochs
        for i in range(n_epochs):
            # enumerate batches over the training set
            for j in range(bat_per_epo):
                # y_real = randint(7, 12, (n_batch, 1)) / 10
                [X_real, labels_real], y_real = self.generate_real_samples(dataset, n_batch)
                [X_fake, labels_fake], y_fake = self.generate_fake_samples(g_model, dataset, n_batch)

                d_loss, _ = d_model.train_on_batch([X_real, X_fake, labels_real, labels_fake], y_real)

                z_input, z_labels, ix = self.generate_latent_points(dataset, n_batch)

                g_loss = gan_model.train_on_batch([dataset[0][ix], z_input, z_labels, z_labels], [y_fake, dataset[0][ix]])
                
                if i % 10 == 0:
                    self.logger.info('>%d, %d/%d, d=(%.3f), g=(%.3f, %.3f)' % (i+1, j+1, bat_per_epo, d_loss, g_loss[0], g_loss[1]))
        # save the generator model
        g_model.save(os.path.join('pretrained', model_name + '.h5'))

#%%
class SRGAN(GAN):
    def create(self, optimizer, losses, loss_weights):
        activation = Dense(1, activation='sigmoid')
        d_model = self.define_discriminator(self.embeddings, activation)
        d_model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        print(d_model.summary())

        g_model = self.define_generator(self.embeddings, self.latent_dim)
        latent_points, labels = g_model.input
        generator_output = g_model.output
        d_model.trainable = False

        discriminator_output = d_model([generator_output, labels])
        # gan_model = Model([latent_points, labels], [discriminator_output])
        gan_model = Model([latent_points, labels], [discriminator_output, generator_output])
        gan_model.compile(loss=losses, loss_weights=loss_weights, optimizer=optimizer)
        print(gan_model.summary())

        return g_model, d_model, gan_model

    def train(self, model_name, g_model, d_model, gan_model, dataset, n_epochs=100, n_batch=2):
        bat_per_epo = int(dataset[0].shape[0] / n_batch)
        half_batch = int(n_batch / 2)
        # manually enumerate epochs
        for i in range(n_epochs):
            # enumerate batches over the training set
            for j in range(bat_per_epo):
                # get randomly selected 'real' samples
                [X_real, labels_real], y_real = self.generate_real_samples(dataset, half_batch)
                [X_fake, labels_fake], y_fake = self.generate_fake_samples(g_model, dataset, half_batch)

                d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)
                d_loss2, _ = d_model.train_on_batch([X_fake, labels_fake], y_fake)

                z_input, z_labels, ix = self.generate_latent_points(dataset, n_batch)
                y_gan = randint(7, 12, (n_batch, 1)) / 10
                # g_loss = gan_model.train_on_batch([z_input, z_labels], [y_gan])
                g_loss = gan_model.train_on_batch([z_input, z_labels], [y_gan, dataset[0][ix]])

                if i % 10 == 0:
                    # self.logger.info('>%d, %d/%d, d=(%.3f, %.3f), g=(%.3f)' % (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
                    self.logger.info('>%d, %d/%d, d=(%.3f, %.3f), g=(%.3f, %.3f)' % (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss[0], g_loss[1]))
        # save the generator model
        g_model.save(os.path.join('pretrained', model_name + '.h5'))
