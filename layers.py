import tensorflow.keras.layers as L

'''
    Implements the layer types of the Steganalysis Residual Network
    presented in @http://www.ws.binghamton.edu/fridrich/Research/SRNet.pdf
'''

def layer_type1(x_inp, filters, stride=1, kernel_size=(3, 3), dropout_rate=0):
    x = L.Conv2D(filters, kernel_size, padding="same", strides=stride)(x_inp)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)
    if dropout_rate > 0:
        x = L.Dropout(dropout_rate)(x)

    return x


def layer_type2(x_inp, filters, kernel_size=(3, 3), dropout_rate=0):
    x = layer_type1(x_inp, filters)
    x = L.Conv2D(filters, kernel_size, padding="same")(x)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)
    if dropout_rate > 0:
        x = L.Dropout(dropout_rate)(x)

    x = L.Add()([x, x_inp])

    return x


def layer_type3(x_inp, filters, kernel_size=(3, 3), dropout_rate=0):
    x = layer_type1(x_inp, filters)
    x = L.Conv2D(filters, kernel_size, padding="same")(x)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)
    x = L.AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    if dropout_rate > 0:
        x = L.Dropout(dropout_rate)(x)

    x_res = L.Conv2D(filters, kernel_size, strides=(2, 2))(x_inp)
    x_res = L.BatchNormalization()(x_res)
    if dropout_rate > 0:
        x_res = L.Dropout(dropout_rate)(x_res)

    x = L.Add()([x, x_res])

    return x

def layer_type4(x_inp, filters, kernel_size=(3, 3), dropout_rate=0):
    x = layer_type1(x_inp, filters)
    x = L.Conv2D(filters, kernel_size=(3, 3), padding="same")(x)
    x = L.BatchNormalization()(x)
    if dropout_rate > 0:
        x = L.Dropout(dropout_rate)(x)
    x = L.Flatten()(x)

    return x
