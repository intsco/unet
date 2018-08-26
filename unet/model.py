from keras.models import Model, Input
from keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler


def unet(pretrained_weights=None, input_size=(256, 256, 1), kernel_initializer='he_normal'):
    inputs = Input(input_size)

    # first encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # second encoder
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # third encoder
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # forth encoder
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # fifth bottom unit
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv5)
    drop5 = Dropout(0.5)(conv5)

    # sixth decoder
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        UpSampling2D(size=(2, 2))(drop5))
    concat6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(concat6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv6)

    # seventh decoder
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        UpSampling2D(size=(2, 2))(conv6))
    concat7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(concat7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv7)

    # eighth decoder
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        UpSampling2D(size=(2, 2))(conv7))
    concat8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(concat8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv8)

    # ninth decoder
    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
        UpSampling2D(size=(2, 2))(conv8))
    concat9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(concat9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv9)

    # tenth one-to-one conv layer
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model
