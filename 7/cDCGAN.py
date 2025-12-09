import os
import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import (
    Dense,
    Reshape,
    Flatten,
    Dropout,
    Input,
    Concatenate,
    BatchNormalization,
    Activation,
    LeakyReLU,
    LSTM,
    Conv1D,
    MaxPooling1D,
    TimeDistributed,
)
from keras.optimizers import RMSprop, Adam
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tensorflow as tf
import time
import pygame
import random

seed = 42
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)
random.seed(seed)


def load_timeseries_data_from_folder(folder, time_steps, feature_dim):
    sequences = []
    for filename in os.listdir(folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder, filename)
            data = pd.read_csv(file_path, header=None).values
            if data.shape == (time_steps, feature_dim):
                sequences.append(data)
    return np.array(sequences)


def build_generator(noise_dim, time_steps, feature_dim, condition_dim):
    noise = Input(shape=(noise_dim,))
    condition = Input(shape=(condition_dim,))
    model_input = Concatenate()([noise, condition])

    x = Dense(512 * time_steps)(model_input)
    x = Reshape((time_steps, 512))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(256, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(128, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = LSTM(256, return_sequences=True)(x)
    x = LSTM(128, return_sequences=True)(x)

    x = TimeDistributed(Dense(feature_dim))(x)
    generated_sequence = Activation("tanh")(x)

    return Model([noise, condition], generated_sequence)


def build_discriminator(time_steps, feature_dim, condition_dim):
    sequence = Input(shape=(time_steps, feature_dim))
    condition = Input(shape=((condition_dim,)))
    condition_reshape = Reshape((time_steps, feature_dim))(condition)
    model_input = Concatenate()([sequence, condition_reshape])

    x = Conv1D(128, kernel_size=3, padding="same")(model_input)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.5)(x)

    x = Conv1D(256, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.5)(x)

    x = Conv1D(512, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.5)(x)

    x = LSTM(512, return_sequences=True)(x)
    x = Dropout(0.5)(x)
    x = LSTM(1024)(x)
    x = Dropout(0.5)(x)

    validity = Dense(1, activation="sigmoid")(x)

    return Model([sequence, condition], validity)


time_steps = 30
feature_dim = 8
noise_dim = 400
condition_dim = time_steps * feature_dim
epochs = 30000
batch_size = 128
save_interval = 100
learning_rate_generator = 0.00005
learning_rate_discriminator = 0.0001
beta_1 = 0.5
music = epochs - 100

hyperparameters_text = """
time_steps = {}
feature_dim = {}
noise_dim = {}
condition_dim = {}
epochs = {}
batch_size = {}
save_interval = {}
learning_rate_generator = {}
learning_rate_discriminator = {}
beta_1 = {}
""".format(
    time_steps,
    feature_dim,
    noise_dim,
    condition_dim,
    epochs,
    batch_size,
    save_interval,
    learning_rate_generator,
    learning_rate_discriminator,
    beta_1,
)

with open("hyperparameters.txt", "w") as file:
    file.write(hyperparameters_text)

a_image_folder = ""
b_image_folder = ""

a_sequences = load_timeseries_data_from_folder(a_image_folder, time_steps, feature_dim)
b_sequences = load_timeseries_data_from_folder(b_image_folder, time_steps, feature_dim)

a_sequences = (a_sequences / 127.5) - 1.0
b_sequences = (b_sequences / 127.5) - 1.0

generator_optimizer = Adam(lr=learning_rate_generator, beta_1=beta_1)
discriminator_optimizer = Adam(lr=learning_rate_discriminator, beta_1=beta_1)

generator = build_generator(noise_dim, time_steps, feature_dim, condition_dim)
discriminator = build_discriminator(time_steps, feature_dim, condition_dim)
discriminator.compile(loss="binary_crossentropy", optimizer=discriminator_optimizer, metrics=["accuracy"])

noise = Input(shape=(noise_dim,))
condition = Input(shape=(condition_dim,))
generated_sequence = generator([noise, condition])

discriminator.trainable = False

validity = discriminator([generated_sequence, condition])
combined = Model([noise, condition], validity)
combined.compile(loss="binary_crossentropy", optimizer=generator_optimizer)

d_losses = []
g_losses = []

fig, ax = plt.subplots()
(line1,) = ax.plot([], [], label="Discriminator loss")
(line2,) = ax.plot([], [], label="Generator loss")
ax.set_xlim(0, epochs)
ax.set_ylim(0, 5)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()

pygame.mixer.init()


def play_music():
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)


def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2


def update(epoch):
    noise = np.random.normal(0, 1, (batch_size, noise_dim))
    idx = np.random.randint(0, b_sequences.shape[0], batch_size)
    conditions = b_sequences[idx].reshape(batch_size, -1)

    gen_sequences = generator.predict([noise, conditions])

    idx = np.random.randint(0, a_sequences.shape[0], batch_size)
    real_sequences = a_sequences[idx]
    real_conditions = b_sequences[np.random.randint(0, b_sequences.shape[0], batch_size)].reshape(batch_size, -1)

    real_labels = np.ones((batch_size, 1)) * 0.9
    fake_labels = np.zeros((batch_size, 1))

    d_loss_real = discriminator.train_on_batch([real_sequences, real_conditions], real_labels)
    d_loss_fake = discriminator.train_on_batch([gen_sequences, conditions], fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    noise = np.random.normal(0, 1, (batch_size, noise_dim))
    valid_y = np.ones((batch_size, 1))
    g_loss = combined.train_on_batch([noise, conditions], valid_y)

    d_losses.append(d_loss[0])
    g_losses.append(g_loss)

    line1.set_data(range(len(d_losses)), d_losses)
    line2.set_data(range(len(g_losses)), g_losses)

    if epoch % save_interval == 0:
        print("epoch {} [D loss: {} | D accuracy: {}%] [G loss: {}]".format(epoch, d_loss[0], d_loss[1] * 100, g_loss))

    if epoch == music:
        play_music()

    return line1, line2


start_time = time.time()

ani = FuncAnimation(fig, update, frames=epochs, init_func=init, blit=True, repeat=False)
plt.show()

fig.savefig("training_loss_graph.png")

generator.save("generator_model.h5")
discriminator.save("discriminator_model.h5")

end_time = time.time()
elapsed_time = end_time - start_time
print("学習所要時間: {:.2f} 分".format(elapsed_time / 60))
