from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, ReLU, Dropout,  GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.initializers import glorot_uniform
import tensorflow as tf


#-----------------------------Resnet50---------------------------+
class ResNet50:
    def identity_block(X, f, filters, stage, block):

        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        
        # Retrieve Filters
        F1, F2, F3 = filters
        
        # Saving the input value.we need this later to add to the output. 
        X_Residual = X
        
        # First component of main path
        X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a')(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        
        # Second component of main path (≈3 lines)
        X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b')(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        # Third component of main path (≈2 lines)
        X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c')(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation 
        X = Add()([X, X_Residual])
        X = Activation('relu')(X)
        
        
        return X
    def convolutional_block(X, f, filters, stage, block, s = 2):
        
        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        
        # Retrieve Filters
        F1, F2, F3 = filters
        
        # Save the input value
        X_Residual = X


        # First layer 
        X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a')(X) # 1,1 is filter size
        X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)  # normalization on channels
        X = Activation('relu')(X)

        
        # Second layer  (f,f)=3*3 filter by default
        X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b')(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
        X = Activation('relu')(X)


        # Third layer
        X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c')(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)


        # Residual Connection
        X_Residual = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1')(X_Residual)
        X_Residual = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_Residual)

        # Final step: Add shortcut value here, and pass it through a RELU activation 
        X = Add()([X, X_Residual])
        X = Activation('relu')(X)
        
        
        return X
    def ResNet50(input_shape, outputs, activation):

        # Define the input as a tensor with shape input_shape
        X_input = Input(input_shape)

        # Zero-Padding
        X = ZeroPadding2D((3, 3))(X_input) #3,3 padding

        # Stage 1
        X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(X) #64 filters of kernel size 7*7 pixels
        X = BatchNormalization(axis=3, name='bn_conv1')(X) #batchnorm applied on channels
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X) #window size is 3*3

        # Stage 2
        X = ResNet50.convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
        X = ResNet50.identity_block(X, 3, [64, 64, 256], stage=2, block='b') 
        X = ResNet50.identity_block(X, 3, [64, 64, 256], stage=2, block='c')

        # Stage 3 
        X = ResNet50.convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
        X = ResNet50.identity_block(X, 3, [128, 128, 512], stage=3, block='b')
        X = ResNet50.identity_block(X, 3, [128, 128, 512], stage=3, block='c')
        X = ResNet50.identity_block(X, 3, [128, 128, 512], stage=3, block='d')

        # Stage 4 
        X = ResNet50.convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
        X = ResNet50.identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
        X = ResNet50.identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
        X = ResNet50.identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
        X = ResNet50.identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
        X = ResNet50.identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

        # Stage 5 
        X = ResNet50.convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
        X = ResNet50.identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
        X = ResNet50.identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

        # AVGPOOL 
        X = AveragePooling2D((2,2), name="avg_pool")(X)

        # output layer
        X = Flatten()(X)
        X = Dense(outputs, activation=activation, name='fc' + str(outputs), kernel_initializer = glorot_uniform(seed=0))(X)
        
        # Create model
        model = Model(inputs = X_input, outputs = X, name='ResNet50')
        
        return model
# ---------------------------- AlexNet---------------------------+
class AlexNet:
    def AlexNet(input_shape, outputs, activation):
        # Input Layer
        X_input = Input(input_shape, name = "AlexNet Input")

        # Layer 1 - Convolutions
        X = Conv2D(filters=96, kernel_size=11, strides=4, padding="same")(X_input)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = MaxPooling2D(pool_size=3, strides=2)(X)

        # Layer 2 - Convolutions
        X = Conv2D(filters=256, kernel_size=5, strides=1, padding="same")(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = MaxPooling2D(pool_size=3, strides=2)(X)

        # Layer 3 - Convolutions
        X = Conv2D(filters=384, kernel_size=3, strides=1, padding="same")(X)
        #X = BatchNormalization()(X)
        X = Activation('relu')(X)

        # Layer 4 - Convolutions
        X = Conv2D(filters=384, kernel_size=3, strides=1, padding="same")(X)
        #X = BatchNormalization()(X)
        X = Activation('relu')(X)

        # Layer 5 - Convolutions
        X = Conv2D(filters=256, kernel_size=3, strides=1, padding="same")(X)
        #X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = MaxPooling2D(pool_size=3, strides=2)(X)

        # Layer 6 - Dense
        X = Flatten()(X)
        X = Dense(units=2048)(X)
        X = Activation('relu')(X)
        X = Dropout(rate=0.5)(X)

        # Layer 7 - Dense
        X = Dense(units=2048)(X)
        X = Activation('relu')(X)
        X = Dropout(rate=0.5)(X)

        # Layer 8 - Dense
        X = Dense(outputs, activation=activation, name='fc' + str(outputs), kernel_initializer = glorot_uniform(seed=0))(X)
        
        # Create model
        model = Model(inputs = X_input, outputs = X, name='AlexNet_model')
        
        return model
# ---------------------------- VGG ------------------------------+
class VGG:
    def VGG16(input_shape, outputs, activation):
        # Determine proper input shape
        X_input = Input(input_shape, name = "VGG16 Input")
        
        # Block 1
        
        X = Conv2D(64, (3, 3),padding='same',name='block1_conv1')(X_input)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(X)

        # Block 2
        X = Conv2D(128, (3, 3),padding='same',name='block2_conv1')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Conv2D(128, (3, 3),padding='same',name='block2_conv2')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(X)

        # Block 3
        X = Conv2D(256, (3, 3),padding='same',name='block3_conv1')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Conv2D(256, (3, 3),padding='same',name='block3_conv2')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Conv2D(256, (3, 3),padding='same',name='block3_conv3')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(X)

        # Block 4
        X = Conv2D(512, (3, 3),padding='same',name='block4_conv1')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Conv2D(512, (3, 3),padding='same',name='block4_conv2')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Conv2D(512, (3, 3),padding='same',name='block4_conv3')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(X)

        # Block 5
        X = Conv2D(512, (3, 3),padding='same', name='block5_conv1')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Conv2D(512, (3, 3),padding='same',name='block5_conv2')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Conv2D(512, (3, 3),padding='same',name='block5_conv3')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(X)
        
        X = Flatten(name='flatten')(X)
        X = Dense(outputs, activation=activation, name='fc_end_' + str(outputs), kernel_initializer = glorot_uniform(seed=0))(X)

        model = Model(inputs = X_input, outputs = X, name='VGG16_model')
        
        return model 
   
    def VGG19(input_shape, outputs, activation):
        # Determine proper input shape
        X_input = Input(input_shape, name = "VGG19 Input")

        # Block 1
        X = Conv2D(64, (3, 3),padding='same',name='block1_conv1')(X_input)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(X)

        # Block 2
        X = Conv2D(128, (3, 3),padding='same',name='block2_conv1')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Conv2D(128, (3, 3),padding='same',name='block2_conv2')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(X)

        # Block 3
        X = Conv2D(256, (3, 3),padding='same',name='block3_conv1')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Conv2D(256, (3, 3),padding='same',name='block3_conv2')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Conv2D(256, (3, 3),padding='same',name='block3_conv4')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(X)

        # Block 4
        X = Conv2D(512, (3, 3),padding='same',name='block4_conv1')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Conv2D(512, (3, 3),padding='same',name='block4_conv2')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Conv2D(512, (3, 3),padding='same',name='block4_conv3')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Conv2D(512, (3, 3),padding='same',name='block4_conv4')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(X)

        # Block 5
        X = Conv2D(512, (3, 3), padding='same', name='block5_conv1')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Conv2D(512, (3, 3),padding='same',name='block5_conv2')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Conv2D(512, (3, 3),padding='same',name='block5_conv3')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Conv2D(512, (3, 3),padding='same',name='block5_conv4')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(X)
        
        X = Flatten(name='flatten')(X)
        X = Dense(outputs, activation=activation, name='fc_end_' + str(outputs), kernel_initializer = glorot_uniform(seed=0))(X)

        model = Model(inputs = X_input, outputs = X, name='VGG19_model')
        
        return model
# ---------------------------- GoogLeNet ------------------------+
class GoogLeNet:
    def inception(x,
              filters_1x1,
              filters_3x3_reduce,
              filters_3x3,
              filters_5x5_reduce,
              filters_5x5,
              filters_pool):
        
        path1 = Conv2D(filters_1x1, (1, 1), padding='same')(x)        
        path1 = BatchNormalization()(path1)
        path1 = Activation('relu')(path1)

        path2 = Conv2D(filters_3x3_reduce, (1, 1), padding='same')(x)
        path2 = BatchNormalization()(path2)
        path2 = Activation('relu')(path2)
        path2 = Conv2D(filters_3x3, (3, 3), padding='same')(path2)
        path2 = BatchNormalization()(path2)
        path2 = Activation('relu')(path2)

        path3 = Conv2D(filters_5x5_reduce, (1, 1), padding='same')(x)
        path3 = BatchNormalization()(path3)
        path3 = Activation('relu')(path3)
        path3 = Conv2D(filters_5x5, (5, 5), padding='same')(path3)
        path3 = BatchNormalization()(path3)
        path3 = Activation('relu')(path3)

        path4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
        path4 = Conv2D(filters_pool, (1, 1), padding='same')(path4)
        path4 = BatchNormalization()(path4)
        path4 = Activation('relu')(path4)

        return tf.concat([path1, path2, path3, path4], axis=3)
    
    def GoogLeNet(input_shape, outputs, activation):
        X_input = Input(input_shape, name = "GoogLeNet Input")
        
        x = Conv2D(64, 7, strides=2, padding='same')(X_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(3, strides=2)(x)

        x = Conv2D(64, 1, strides=1, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(192, 3, strides=1, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = MaxPooling2D(3, strides=2)(x)

        x = GoogLeNet.inception(x,
                    filters_1x1=64,
                    filters_3x3_reduce=96,
                    filters_3x3=128,
                    filters_5x5_reduce=16,
                    filters_5x5=32,
                    filters_pool=32)

        x = GoogLeNet.inception(x,
                    filters_1x1=128,
                    filters_3x3_reduce=128,
                    filters_3x3=192,
                    filters_5x5_reduce=32,
                    filters_5x5=96,
                    filters_pool=64)

        x = MaxPooling2D(3, strides=2)(x)

        x = GoogLeNet.inception(x,
                    filters_1x1=192,
                    filters_3x3_reduce=96,
                    filters_3x3=208,
                    filters_5x5_reduce=16,
                    filters_5x5=48,
                    filters_pool=64)

        aux1 = AveragePooling2D((5, 5), strides=3)(x)
        aux1 = Conv2D(128, 1, padding='same')(aux1)
        aux1 = BatchNormalization()(aux1)
        aux1 = Activation('relu')(aux1)
        aux1 = Flatten()(aux1)
        aux1 = Dense(1024, activation='relu')(aux1)
        aux1 = Dropout(0.5)(aux1)
        aux1 = Dense(1024, activation='relu')(aux1)
        aux1 = Dropout(0.5)(aux1)
        aux1 = Dense(outputs, activation=activation)(aux1)

        x = GoogLeNet.inception(x,
                    filters_1x1=160,
                    filters_3x3_reduce=112,
                    filters_3x3=224,
                    filters_5x5_reduce=24,
                    filters_5x5=64,
                    filters_pool=64)

        x = GoogLeNet.inception(x,
                    filters_1x1=128,
                    filters_3x3_reduce=128,
                    filters_3x3=256,
                    filters_5x5_reduce=24,
                    filters_5x5=64,
                    filters_pool=64)

        x = GoogLeNet.inception(x,
                    filters_1x1=112,
                    filters_3x3_reduce=144,
                    filters_3x3=288,
                    filters_5x5_reduce=32,
                    filters_5x5=64,
                    filters_pool=64)

        aux2 = AveragePooling2D((5, 5), strides=3)(x)
        aux2 = Conv2D(128, 1, padding='same')(aux2)
        aux2 = BatchNormalization()(aux2)
        aux2 = Activation('relu')(aux2)
        aux2 = Flatten()(aux2)
        aux2 = Dense(1024, activation='relu')(aux2)
        aux2 = Dropout(0.5)(aux2)
        aux2 = Dense(1024, activation='relu')(aux2)
        aux2 = Dropout(0.5)(aux2)
        aux2 = Dense(outputs, activation=activation)(aux2)

        x = GoogLeNet.inception(x,
                    filters_1x1=256,
                    filters_3x3_reduce=160,
                    filters_3x3=320,
                    filters_5x5_reduce=32,
                    filters_5x5=128,
                    filters_pool=128)

        x = MaxPooling2D(3, strides=2)(x)

        x = GoogLeNet.inception(x,
                    filters_1x1=256,
                    filters_3x3_reduce=160,
                    filters_3x3=320,
                    filters_5x5_reduce=32,
                    filters_5x5=128,
                    filters_pool=128)

        x = GoogLeNet.inception(x,
                    filters_1x1=384,
                    filters_3x3_reduce=192,
                    filters_3x3=384,
                    filters_5x5_reduce=48,
                    filters_5x5=128,
                    filters_pool=128)

        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        out = Dense(outputs, activation=activation)(x)
        
        model = Model(inputs = X_input, outputs = [out, aux1, aux2], name='GoogLeNet')
        
        return model 