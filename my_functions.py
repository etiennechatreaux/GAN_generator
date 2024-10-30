import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam

from typing import Optional


def define_generator(latent_dim: int, add_layer: Optional[bool] = False, alpha: float = 0.2, n_outputs: int = 30) -> Sequential:
    model = Sequential()
    model.add(Dense(15, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=alpha))
    model.add(BatchNormalization())
    model.add(Dense(30))
    if add_layer:
        model.add(LeakyReLU(alpha=alpha))
        model.add(BatchNormalization())
        model.add(Dense(30))
    model.add(LeakyReLU(alpha=alpha))
    model.add(BatchNormalization())
    model.add(Dense(n_outputs, activation='tanh'))
    return model

def define_discriminator(n_inputs: int = 2, add_layer: bool = False, alpha: float = 0.2, lr: float = 0.0002) -> Sequential:
    model = Sequential()
    model.add(Dense(25, input_dim=n_inputs))
    model.add(LeakyReLU(alpha=alpha))
    if add_layer:
        model.add(LeakyReLU(alpha=alpha))
        model.add(BatchNormalization())
        model.add(Dense(30))
    model.add(Dense(50))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr, beta_1=0.5))
    return model

def define_gan(generator: Sequential, discriminator: Sequential, loss: str = 'binary_crossentropy', lr: float = 0.0002, beta_1: float = 0.5) -> Sequential:
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss=loss, optimizer=Adam(lr, beta_1))
    return model

def generate_noise(n_samples: int, latent_dim: int) -> np.ndarray:
    noise_matrix = np.random.randn(latent_dim * n_samples)
    noise_matrix = noise_matrix.reshape(n_samples, latent_dim)
    return noise_matrix

def train_gan(generator: Sequential, discriminator: Sequential, gan: Sequential, dataset: pd.DataFrame, latent_dim: int = 30, epochs: int = 500, batch_size: int = 64) -> tuple:
    d_losses = [] 
    g_losses = []
    half_batch = int(batch_size / 2)

    for e in range(epochs):
        noise = generate_noise(half_batch, dataset.shape[1])
        generated_data = generator.predict(noise)
        real_data = dataset.sample(half_batch)

        X = np.concatenate([real_data, generated_data])
        y = np.ones(2 * batch_size)

        y = np.zeros(2 * half_batch)
        y[:half_batch] = 1

        d_loss = discriminator.train_on_batch(X, y)
        d_losses.append(d_loss)

        noise = generate_noise(batch_size, latent_dim)
        y_gan = np.ones(batch_size)
        g_loss = gan.train_on_batch(noise, y_gan)
        g_losses.append(g_loss)

        gan.train_on_batch(noise, y_gan)
        if e % 50 == 0:
            print(f"E:{e}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}")
    print("Entrainement fini :", d_losses[-1], g_losses[-1])
    display_losses(d_losses, g_losses)
    return d_losses[-1], g_losses[-1]

def display_losses(d_losses, g_losses):
    plt.figure(figsize=(12, 6))
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.title('GAN Training Losses')
    plt.xlabel('Epochs (Batches)')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_generated_data(generator, latent_dim, examples=10):
    noise = np.random.randn(examples, latent_dim)
    generated_data = generator.predict(noise)
    plt.scatter(generated_data[:, 0], generated_data[:, 1])
    plt.title("Generated Data")
    plt.show()