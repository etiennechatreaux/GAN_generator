from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import numpy as np




def build_discriminator(img_shape):
    model = models.Sequential()
    
    model.add(layers.Conv2D(64, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model

def build_generator(latent_dim):
    model = models.Sequential()

    model.add(layers.Dense(128 * 64 * 64, activation='relu', input_dim=latent_dim))
    model.add(layers.Reshape((64, 64, 128)))
    
    model.add(layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(3, kernel_size=3, activation='tanh', padding='same'))

    return model








def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = models.Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

def train_gan(generator, discriminator, combined_models, train_set, epochs=10000, batch_size=32, modulo_epoch_display=100):
    half_batch = batch_size // 2
    
    for epoch in range(epochs):
        # Entraînement du discriminateur
        idx = np.random.randint(0, train_set.shape[0], half_batch)
        real_imgs = train_set[idx]
        
        noise = np.random.normal(0, 1, (half_batch, 100))  # Bruit aléatoire
        gen_imgs = generator.predict(noise)
        
        d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Entraînement du générateur
        noise = np.random.normal(0, 1, (batch_size, 100))
        valid_y = np.ones((batch_size, 1))
        g_loss = combined_models.train_on_batch(noise, valid_y)
        
        # Affichage des pertes
        if epoch % modulo_epoch_display == 0:
            print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}] [G loss: {g_loss}]")