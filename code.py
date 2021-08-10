import random
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import librosa

# Set seeds
SEED = 21
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Load data
print('Loading data...')


def read_data(file):
    data = np.load(file, allow_pickle=True)
    return data[0] if len(data) == 1 else data


data, labels = read_data('data/raw_data.pkl')
test_data = read_data('data/test_data.pkl')

TEST_SIZE = test_data.shape[0]
NUM_PERSONS = len(np.unique(labels))
OBS_PER_PERSON = int(len(data) / NUM_PERSONS)
SAMPLE_RATE = data.shape[1]

# Feature engineering
print('Extracting features...')


def audio_features(y):
    sr = 11025
    n_fft = 1024
    hop_length = n_fft // 2
    n_mfcc = 128
    fmin = 16

    stft = np.abs(librosa.stft(y, n_fft=n_fft))
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                         n_fft=n_fft, hop_length=hop_length, fmin=fmin).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(
        S=stft, sr=sr, n_fft=n_fft, hop_length=hop_length).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(
        y, sr=sr, fmin=fmin).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(
        S=stft, sr=sr, fmin=fmin).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(
        y=librosa.effects.harmonic(y), sr=sr).T, axis=0)

    features = np.concatenate([mfccs, chroma, mel, contrast, tonnetz])
    return features


features_list = []
for row in data:
    features = audio_features(row)
    features_list.append(features)
features = np.array(features_list)

# Preprare train data
print('Preparing data for training...')


def prepare_train_data(features, labels):
    X_list, y_list = [], []
    for person_id in np.unique(labels):
        same_person = features[labels == person_id]
        other_person = features[labels != person_id]
        other_person_random = other_person[np.random.randint(
            other_person.shape[0], size=OBS_PER_PERSON**2)]

        num = 0
        for row_1 in same_person:
            for row_2 in same_person:
                X_list.append(np.concatenate((row_1, row_2)))
                y_list.append(1)
                X_list.append(np.concatenate(
                    (row_1, other_person_random[num])))
                y_list.append(0)
                X_list.append(np.concatenate(
                    (other_person_random[num], row_1)))
                y_list.append(0)
                num += 1

    X, y = np.array(X_list), np.array(y_list)
    y = tf.keras.utils.to_categorical(y, num_classes=2, dtype='float32')
    return X, y


X, y = prepare_train_data(features, labels)


# Train model
print('Training model...')
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, shuffle=True)
N_DIM = X_train.shape[1]


def build_model():
    model = tf.keras.Sequential([
        Input(shape=N_DIM),
        BatchNormalization(),
        Dense(512),
        LeakyReLU(),
        Dense(256),
        LeakyReLU(),
        Dropout(0.3, seed=SEED),
        Dense(128),
        LeakyReLU(),
        Dense(64),
        Dropout(0.3, seed=SEED),
        LeakyReLU(),
        Dense(32),
        LeakyReLU(),
        Dense(16),
        LeakyReLU(),
        Dense(8),
        LeakyReLU(),
        Dense(2, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


model = build_model()


early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', patience=3, factor=0.1, min_lr=1e-7)
history = model.fit(X_train,
                    y_train,
                    batch_size=32,
                    epochs=50,
                    callbacks=[early_stopping, reduce_lr],
                    validation_data=(X_val, y_val))

test_loss, test_acc = model.evaluate(X_val, y_val, verbose=0)
print(f'- Model accuracy: {test_acc:.4f}')

# Prepare test data
print('Preparing test set...')
test_features_list = []
for row in test_data:
    test_features = audio_features(row)
    test_features_list.append(test_features)
test_features = np.array(test_features_list)

X_test_list = []
for row_1 in test_features:
    for row_2 in test_features:
        X_test_list.append(np.concatenate((row_1, row_2)))
X_test = np.array(X_test_list)
X_test.shape

# Predict
print('Making prediction...')
y_pred = model.predict(X_test)
distance_matrix = y_pred[:, 0].reshape(TEST_SIZE, TEST_SIZE)

np.savetxt('submission/answer.txt',
           distance_matrix, delimiter=';', fmt='%.12f')

print('Done.')
