import os
import numpy as np
import tensorflow as tf

# This is for parsing the X data, you can ignore it if you do not need preprocessing
def format_data_x(datafile):
    x_data = None
    for item in datafile:
        item_data = np.loadtxt(item, dtype=float)
        if x_data is None:
            x_data = np.zeros((len(item_data), 1))
        x_data = np.hstack((x_data, item_data))
    x_data = x_data[:, 1:]
    print(x_data.shape)
    X = None
    for i in range(len(x_data)):
        row = np.asarray(x_data[i, :])
        row = row.reshape(9, 128).T
        if X is None:
            X = np.zeros((len(x_data), 128, 9))
        X[i] = row
    print(X.shape)
    return X

# This is for parsing the Y data, you can ignore it if you do not need preprocessing
def format_data_y(datafile):
    data = np.loadtxt(datafile, dtype=int) - 1
    YY = np.eye(6)[data]
    return YY

# Load data function, if there exists parsed data file, then use it
# If not, parse the original dataset from scratch
def load_data():
    if os.path.isfile('data/data_har.npz'):
        data = np.load('data/data_har.npz')
        X_train = data['X_train']
        Y_train = data['Y_train']
        X_test = data['X_test']
        Y_test = data['Y_test']
    else:
        # This for processing the dataset from scratch
        # After downloading the dataset, put it to somewhere that str_folder can find
        str_folder = '/Users/dhamo_85/Downloads/UCI HAR Dataset/'  # Update this path
        INPUT_SIGNAL_TYPES = [
            "body_acc_x_",
            "body_acc_y_",
            "body_acc_z_",
            "body_gyro_x_",
            "body_gyro_y_",
            "body_gyro_z_",
            "total_acc_x_",
            "total_acc_y_",
            "total_acc_z_"
        ]

        str_train_files = [str_folder + 'train/' + 'Inertial Signals/' + item + 'train.txt' for item in INPUT_SIGNAL_TYPES]
        str_test_files = [str_folder + 'test/' + 'Inertial Signals/' + item + 'test.txt' for item in INPUT_SIGNAL_TYPES]
        str_train_y = str_folder + 'train/y_train.txt'
        str_test_y = str_folder + 'test/y_test.txt'

        X_train = format_data_x(str_train_files)
        X_test = format_data_x(str_test_files)
        Y_train = format_data_y(str_train_y)
        Y_test = format_data_y(str_test_y)

    return X_train, Y_train, X_test, Y_test

# A class for some hyperparameters
class Config(object):
    def __init__(self, X_train, Y_train):
        self.n_input = len(X_train[0])  # number of input neurons to the network
        self.n_output = len(Y_train[0])  # number of output neurons
        self.dropout = 0.8  # dropout, between 0 and 1
        self.learning_rate = 0.001  # learning rate, float
        self.training_epoch = 20  # training epoch
        self.n_channel = 9  # number of input channel
        self.input_height = 128  # input height
        self.input_width = 1  # input width
        self.kernel_size = 64  # number of convolution kernel size
        self.depth = 32  # number of convolutions
        self.batch_size = 16  # batch size
        self.show_progress = 50  # how many batches to show the progress

# wrap of conv1d
def conv1d(x, W, b, stride):
    x = tf.nn.conv2d(x, W, strides=[1, stride, 1, 1], padding='SAME')
    x = tf.add(x, b)
    return tf.nn.relu(x)

# wrap of maxpool1d
def maxpool1d(x, kernel_size, stride):
    return tf.nn.max_pool(x, ksize=[1, kernel_size, 1, 1], strides=[1, stride, 1, 1], padding='VALID')

# network definition
class ConvNet(tf.keras.Model):
    def __init__(self, config):
        super(ConvNet, self).__init__()
        self.config = config
        self.conv1 = tf.keras.layers.Conv2D(filters=config.depth, kernel_size=(1, config.kernel_size), strides=(1, 1), padding='same', activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='valid')
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, config.kernel_size), strides=(1, 1), padding='same', activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='valid')
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(units=1000, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(rate=1-config.dropout)
        self.fc2 = tf.keras.layers.Dense(units=500, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(rate=1-config.dropout)
        self.fc3 = tf.keras.layers.Dense(units=300, activation='relu')
        self.dropout3 = tf.keras.layers.Dropout(rate=1-config.dropout)
        self.out = tf.keras.layers.Dense(units=config.n_output, activation=None)

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x, training=training)
        x = self.fc2(x)
        x = self.dropout2(x, training=training)
        x = self.fc3(x)
        x = self.dropout3(x, training=training)
        x = self.out(x)
        return x

# wrap the network for training and testing
def network(X_train, Y_train, X_test, Y_test):
    config = Config(X_train, Y_train)

    model = ConvNet(config)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(X_train, Y_train, epochs=config.training_epoch, batch_size=config.batch_size, verbose=1)

    loss, acc = model.evaluate(X_test, Y_test)
    print('Test accuracy: %.8f' % acc)

if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = load_data()
    X_train = (X_train - np.mean(X_train)) / np.std(X_train)
    X_test = (X_test - np.mean(X_test)) / np.std(X_test)
    X_train = np.reshape(X_train, (-1, 128, 1, 9))
    X_test = np.reshape(X_test, (-1, 128, 1, 9))
    network(X_train, Y_train, X_test, Y_test)
