# FileName:ANN_main.py
# coding = utf-8
# Created by Hzq
from ANN_utils import *


def s_1():
    train_x = np.linspace(0, 2*np.pi, 9)
    train_y = np.sin(train_x)
    test_x = np.linspace(0, 2*np.pi, 361)
    test_y = np.sin(test_x)

    model = MlpNn(train_x.shape[1:])
    model.add_layer(64, 'tanh')
    model.add_layer(1, 'linear')
    model.compile()
    model.fit(train_x, train_y, epoch=20000,
              optimizer='adam', learn_rate=0.001, verbose=2)
    model.show_predict2D(test_x, test_y)


def s_2():
    train_x = np.linspace(0, 2*np.pi, 9)
    train_y = np.abs(np.sin(train_x))
    test_x = np.linspace(0, 2*np.pi, 361)
    test_y = np.abs(np.sin(test_x))

    model = MlpNn(train_x.shape[1:])
    model.add_layer(64, 'tanh')
    model.add_layer(64, 'tanh')
    model.add_layer(32, 'tanh')
    model.add_layer(32, 'tanh')
    model.add_layer(16, 'tanh')
    model.add_layer(16, 'tanh')
    model.add_layer(1, 'linear')
    model.compile()
    model.fit(train_x, train_y, epoch=50000,
              optimizer='adam', learn_rate=0.003, verbose=2)
    model.show_predict2D(test_x, test_y)


def s_3():
    train_x = np.random.uniform(-5, 5, (128, 2))
    train_y = 100 * np.square(train_x[:, 1] - np.square(train_x[:, 0])) + np.square(1 - train_x[:, 0])
    test_x = np.random.uniform(-5, 5, (1024, 2))
    test_y = 100 * np.square(test_x[:, 1] - np.square(test_x[:, 0])) + np.square(1 - test_x[:, 0])
    train_x, x_off, x_mul = normalization(train_x)
    train_y, y_off, y_mul = normalization(train_y)
    test_x, _, _ = normalization(test_x, x_off, x_mul)

    model = MlpNn(train_x.shape[1:])
    model.add_layer(64, 'sigmoid')
    model.add_layer(64, 'sigmoid')
    model.add_layer(32, 'sigmoid')
    model.add_layer(32, 'sigmoid')
    model.add_layer(16, 'sigmoid')
    model.add_layer(16, 'sigmoid')
    model.add_layer(1, 'linear')
    model.compile()
    model.fit(train_x, train_y, epoch=8000,
              optimizer='adam', learn_rate=0.001, verbose=2)
    model.show_predict3D(test_x, test_y, x_offset=x_off, x_mul=x_mul, y_offset=y_off, y_mul=y_mul)


s_1()

