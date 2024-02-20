import tensorflow as tf
from tensorflow.keras import layers

class ConvBlock2D(tf.keras.Model):
    def __init__(self, filters, kernel_size=1, strides=1, padding='valid', use_relu=True):
        super(ConvBlock2D, self).__init__()
        self.conv = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU() if use_relu else layers.Activation('linear')

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ResidualBlock2D(tf.keras.Model):
    def __init__(self, filters, opt, strides=1):
        super(ResidualBlock2D, self).__init__()
        self.opt = opt

        if self.opt == 0:
            self.conv1_3 = ConvBlock2D(filters, kernel_size=(3, 1), strides=(strides, 1), padding='same')
            self.conv2_3 = ConvBlock2D(filters, kernel_size=(3, 1), strides=(1, 1), padding='same', use_relu=False)

        elif self.opt < 4:
            self.conv1_3 = ConvBlock2D(filters, kernel_size=(3, 1), strides=(strides, 1), padding='same')
            self.conv2_3 = ConvBlock2D(filters, kernel_size=(3, 1), strides=(1, 1), padding='same', use_relu=False)

            self.conv1_5 = ConvBlock2D(filters, kernel_size=(5, 1), strides=(strides, 1), padding='same')
            self.conv2_5 = ConvBlock2D(filters, kernel_size=(5, 1), strides=(1, 1), padding='same', use_relu=False)

            if self.opt == 1 or self.opt == 2:
                self.conv1_1 = ConvBlock2D(filters, kernel_size=1, strides=(strides, 1))
                self.conv2_1 = ConvBlock2D(filters, kernel_size=1, strides=(1, 1), use_relu=False)

        else:
            self.conv1_1 = ConvBlock2D(filters, kernel_size=1, strides=(strides, 1), use_relu=False)

            self.conv0_3 = ConvBlock2D(filters//2, kernel_size=1, strides=(strides, 1))
            self.conv1_3 = ConvBlock2D(filters//2, kernel_size=(3, 1), strides=(strides, 1), padding='same')
            self.conv2_3 = ConvBlock2D(filters, kernel_size=1, strides=(1, 1), use_relu=False)

            self.conv0_5 = ConvBlock2D(filters//2, kernel_size=1, strides=(strides, 1))
            self.conv1_5 = ConvBlock2D(filters//2, kernel_size=(5, 1), strides=(strides, 1), padding='same')
            self.conv2_5 = ConvBlock2D(filters, kernel_size=1, strides=(1, 1), use_relu=False)

        self.shortcut = tf.identity
        if strides != 1:
            self.shortcut = ConvBlock2D(filters, kernel_size=1, strides=(strides, 1), use_relu=False)

    def call(self, x):
        residual = self.shortcut(x)

        if self.opt == 0:
            out = self.conv1_3(x)
            out = self.conv2_3(out)

        elif self.opt < 4:
            out_3 = self.conv1_3(x)
            out_3 = self.conv2_3(out_3)

            out_5 = self.conv1_5(x)
            out_5 = self.conv2_5(out_5)

            if self.opt != 3:
                out_1 = self.conv1_1(x)
                out_1 = self.conv2_1(out_1)
                out = out_1 + out_3 + out_5
            else:
                out = out_3 + out_5

        else:
            out_1 = self.conv1_1(x)

            out_3 = self.conv0_3(x)
            out_3 = self.conv1_3(out_3)
            out_3 = self.conv2_3(out_3)

            out_5 = self.conv0_5(x)
            out_5 = self.conv1_5(out_5)
            out_5 = self.conv2_5(out_5)

            out = out_1 + out_3 + out_5

        out = tf.nn.relu(out + residual)
        return out

class ResNet(tf.keras.Model):
    def __init__(self, num_channels, num_classes, num_resblocks, opt):
        super(ResNet, self).__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_resblocks = num_resblocks
        self.opt = opt
        self.num_filters = 64

        self.conv1 = ConvBlock2D(self.num_filters, kernel_size=(7, 1), strides=(2, 1), padding='same', use_relu=True)
        self.maxpool = layers.MaxPooling2D(pool_size=(3, 1), strides=(2, 1), padding='same')

        self.conv2 = ConvBlock2D(self.num_filters, kernel_size=(3, 1), strides=(2, 1), padding='same', use_relu=True)
        self.conv3 = ConvBlock2D(self.num_filters, kernel_size=(3, 1), strides=(2, 1), padding='same', use_relu=True)

        self.resblocks = [ResidualBlock2D(self.num_filters, self.opt) for _ in range(self.num_resblocks)]
        self.avgpool2 = layers.GlobalAveragePooling2D()
        self.dropout = layers.Dropout(0.2)
        self.fc = layers.Dense(self.num_classes, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)

        for resblock in self.resblocks:
            x = resblock(x)

        x = self.avgpool2(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x