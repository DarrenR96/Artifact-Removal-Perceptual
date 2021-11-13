# Work in progress

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from keras.utils.vis_utils import plot_model

class CNNBNReluDown(layers.Layer):
    def __init__(self, numFilters, size, strides, bn=False):
        super(CNNBNReluDown, self).__init__()
        self.numFilters = numFilters
        self.size = size
        self.bn = bn
        self.strides = strides
        self.convLayer = layers.Conv2D(self.numFilters, self.size, strides=self.strides ,padding="same")
        if self.bn:
            self.bnLayer = layers.BatchNormalization()
        self.reluLayer = layers.LeakyReLU()
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'numFilters' : self.numFilters,
            'size' : self.size,
            'bn' : self.bn
        })
        return config
    
    def call(self, inputs, training=False):
        x = self.convLayer(inputs)
        if self.bn:
            x = self.bnLayer(x, training=training)
        x = self.reluLayer(x)
        return x


class CNNBNReluUp(layers.Layer):
    def __init__(self, numFilters, size, strides, dropout=False):
        super(CNNBNReluUp, self).__init__()
        self.numFilters = numFilters
        self.size = size
        self.dropout = dropout
        self.strides = strides
        self.convLayer = layers.Conv2DTranspose(self.numFilters, self.size, strides=self.strides, padding="same")
        if self.dropout:
            self.dropoutLayer = layers.Dropout(0.3)
        self.reluLayer = layers.LeakyReLU()
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'numFilters' : self.numFilters,
            'size' : self.size,
            'dropout' : self.dropout
        })
        return config
    
    def call(self, inputs, training=False):
        x = self.convLayer(inputs)
        if self.dropout:
            x = self.dropoutLayer(x, training=training)
        x = self.reluLayer(x)
        return x

# encoder = ((numFilters,size,bn) ....  (numFilters,size,bn))

class SpatialSuppression(keras.Model):
    def __init__(self, encoder, decoder, outChannels = 3):
        super(SpatialSuppression, self).__init__()
        self.numEncoderBlocks = len(encoder)
        self.numDecoderBlocks = len(decoder)
        self.encoder = []
        self.decoder = []
        for encoder_opts in encoder:
            self.encoder.append(CNNBNReluDown(encoder_opts[0],encoder_opts[1],encoder_opts[2]))
        
        for decoder_opts in decoder:
            self.decoder.append(CNNBNReluUp(decoder_opts[0],decoder_opts[1],decoder_opts[2]))
        self.lastConv = layers.Conv2DTranspose(outChannels, 4, strides=2, padding="same")
    
    def call(self, x, training=False):
        
        skips = []
        for encoderLayer in self.encoder:
            x = encoderLayer(x, training=training)
            skips.append(x)
        
        skips = reversed(skips[:-1])
        
        for decoderLayer, skip in zip(self.decoder, skips):
            x = decoderLayer(x, training=training)
            x = layers.Concatenate()([x, skip])
       
        x = self.lastConv(x)
        
        return x
    
    def model(self):
        x = keras.Input(shape=(1080,1920,3))
        return keras.Model(inputs=[x], outputs=self.call(x))

class SpatialSuppression(keras.Model):
    def __init__(self, encoder, decoder, outChannels = 3):
        super(SpatialSuppression, self).__init__()
        self.numEncoderBlocks = len(encoder)
        self.numDecoderBlocks = len(decoder)
        self.encoder = []
        self.decoder = []
        for encoder_opts in encoder:
            self.encoder.append(CNNBNReluDown(encoder_opts[0],encoder_opts[1],encoder_opts[2], encoder_opts[3]))
        
        for decoder_opts in decoder:
            self.decoder.append(CNNBNReluUp(decoder_opts[0],decoder_opts[1],decoder_opts[2],decoder_opts[3]))
        self.lastConv = layers.Conv2DTranspose(outChannels, 4, strides=2, padding="same")
    
    def call(self, x, training=False):
        
        skips = []
        for encoderLayer in self.encoder:
            x = encoderLayer(x, training=training)
            skips.append(x)
        
        skips = reversed(skips[:-1])
        
        for decoderLayer, skip in zip(self.decoder, skips):
            x = decoderLayer(x, training=training)
            x = layers.Concatenate()([x, skip])
       
        x = self.lastConv(x)
        
        return x
    
    def model(self):
        x = keras.Input(shape=(1080,1920,3))
        return keras.Model(inputs=[x], outputs=self.call(x))
    

class CNN3D_BNReluDown(layers.Layer):
    def __init__(self, numFilters, size, strides, bn=False):
        super(CNN3D_BNReluDown, self).__init__()
        self.numFilters = numFilters
        self.size = size
        self.bn = bn
        self.strides = strides
        self.conv3DLayer = layers.Conv3D(self.numFilters, self.size, strides=self.strides ,padding="same")
        if self.bn:
            self.bnLayer = layers.BatchNormalization()
        self.reluLayer = layers.LeakyReLU()
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'numFilters' : self.numFilters,
            'size' : self.size,
            'bn' : self.bn
        })
        return config
    
    def call(self, inputs, training=False):
        x = self.conv3DLayer(inputs)
        if self.bn:
            x = self.bnLayer(x, training=training)
        x = self.reluLayer(x)
        return x


class CNN3D_BNReluUp(layers.Layer):
    def __init__(self, numFilters, size, strides, dropout=False):
        super(CNN3D_BNReluUp, self).__init__()
        self.numFilters = numFilters
        self.size = size
        self.dropout = dropout
        self.strides = strides
        self.conv3DLayer = layers.Conv3DTranspose(self.numFilters, self.size, strides=self.strides, padding="same")
        if self.dropout:
            self.dropoutLayer = layers.Dropout(0.3)
        self.reluLayer = layers.LeakyReLU()
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'numFilters' : self.numFilters,
            'size' : self.size,
            'dropout' : self.dropout
        })
        return config
    
    def call(self, inputs, training=False):
        x = self.conv3DLayer(inputs)
        if self.dropout:
            x = self.dropoutLayer(x, training=training)
        x = self.reluLayer(x)
        return x

class TemporalSuppression(keras.Model):
    def __init__(self, encoder, decoder, outChannels = 3):
        super(TemporalSuppression, self).__init__()
        self.numEncoderBlocks = len(encoder)
        self.numDecoderBlocks = len(decoder)
        self.encoder = []
        self.decoder = []
        for encoder_opts in encoder:
            self.encoder.append(CNN3D_BNReluDown(encoder_opts[0],encoder_opts[1],encoder_opts[2],encoder_opts[3]))
        
        for decoder_opts in decoder:
            self.decoder.append(CNN3D_BNReluUp(decoder_opts[0],decoder_opts[1],decoder_opts[2],decoder_opts[3]))
        self.lastConv = layers.Conv3DTranspose(outChannels, 4, strides=(1,2,2), padding="same")
    
    def call(self, x, training=False):
        
        skips = []
        for encoderLayer in self.encoder:
            x = encoderLayer(x, training=training)
            skips.append(x)
        
        skips = reversed(skips[:-1])
        
        for decoderLayer, skip in zip(self.decoder, skips):
            x = decoderLayer(x, training=training)
            x = layers.Concatenate()([x, skip])
       
        x = self.lastConv(x, training=training)
        
        return x
    
    def model(self):
        x = keras.Input(shape=(3,1080,1920,3))
        return keras.Model(inputs=[x], outputs=self.call(x))

class VideoQualityAssessment(keras.Model):
    def __init__(self, spatialBlocks, temporalBlock, finalBlock, denseBlock):
        super(VideoQualityAssessment, self).__init__()
        self.spatialBlocks = []
        self.temporalBlock = []
        self.finalBlock = []
        self.denseBlock = []
        
        for spatial in spatialBlocks:
            self.spatialBlocks.append(CNNBNReluDown(spatial[0],spatial[1],spatial[2], spatial[3]))
        
        for temporal in temporalBlock:
            self.temporalBlock.append(CNNBNReluDown(temporal[0],temporal[1],temporal[2], temporal[3]))
        
        for final in finalBlock:
            self.finalBlock.append(CNNBNReluDown(final[0],final[1],final[2], final[3]))
        
        for dense in denseBlock:
            self.denseBlock.append(layers.Dense(dense))
    
    def call(self, x_ref_min1, x_ref, x_ref_pl1, x_dist_min1, x_dist, x_dist_pl1, training = False):
        for spatial in self.spatialBlocks:
            x_ref_min1 = spatial(x_ref_min1, training=training)
        for spatial in self.spatialBlocks:
            x_ref = spatial(x_ref, training=training)
        for spatial in self.spatialBlocks:
            x_ref_pl1 = spatial(x_ref_pl1, training=training)
            
        x_ref = layers.Concatenate()([x_ref_min1, x_ref, x_ref_pl1])
        
        for spatial in self.spatialBlocks:
            x_dist_min1 = spatial(x_dist_min1, training=training)
        for spatial in self.spatialBlocks:
            x_dist = spatial(x_dist, training=training)
        for spatial in self.spatialBlocks:
            x_dist_pl1 = spatial(x_dist_pl1, training=training)
        
        x_dist = layers.Concatenate()([x_dist_min1, x_dist, x_dist_pl1])
        
                
        for temporal in self.temporalBlock:
            x_ref = temporal(x_ref, training=training)
        for temporal in self.temporalBlock:
            x_dist = temporal(x_dist, training=training)
            
        x = layers.Concatenate()([x_ref, x_dist])
        
        for final in self.finalBlock:
            x = final(x, training=training)
        
        x = layers.Flatten()(x)
        
        for dense in self.denseBlock:
            x = dense(x, training=training)
            
        return x
        
    def model(self):
        x_ref_min1 = keras.Input(shape=(1080,1920,3))
        x_ref = keras.Input(shape=(1080,1920,3))
        x_ref_pl1 = keras.Input(shape=(1080,1920,3))

        x_dist_min1 = keras.Input(shape=(1080,1920,3))
        x_dist = keras.Input(shape=(1080,1920,3))
        x_dist_pl1 = keras.Input(shape=(1080,1920,3))
        
        return keras.Model(inputs=[x_ref_min1, x_ref, x_ref_pl1, x_dist_min1, x_dist, x_dist_pl1], 
                           outputs=self.call(x_ref_min1, x_ref, x_ref_pl1, x_dist_min1, x_dist, x_dist_pl1))
        

        
 spatialBlock = [
    (64, 3, 2, True),
    (64, 3, 2, False),
    (128, 3, 2, False),
    (128, 3, 2, False),
]

temporalBlock = [
    (128, 3, 2, False),
    (128, 3, 2, False),
    (256, 3, 2, False),
]

finalBlock = [
    (256, 3, 2, False),
    (256, 3, 2, False),
    (512, 3, 2, False),
]

denseBlock = [
    1024,
    512,
    128,
    1
]

model = VideoQualityAssessment(spatialBlock,temporalBlock, finalBlock, denseBlock).model()
model.summary()



encoder = [
    (64, 4, (1,2,2), True),
    (64, 4, (1,2,2), False),
    (128, 4, (1,2,2), False),
    (128, 5, (1,5,5), False),
    (256, 4, (1,3,3), False),
    (256, 4, (1,3,2), False),
    (256, 3, (1,3,2), False),
    (512, 3, (1,2,2), False),
]

decoder = [
    (256, 3, (1,1,2), True),
    (256, 4, (1,3,2), True),
    (256, 4, (1,3,2), True),
    (128, 4, (1,3,3), False),
    (128, 5, (1,5,5), False),
    (64, 4, (1,2,2), False),
    (64, 4, (1,2,2), False),
]
    
model = TemporalSuppression(encoder,decoder).model()
model.summary()


encoder = [
    (64, 4, 2, True),
    (64, 4, 2, False),
    (128, 4, 2, False),
    (128, 5, 5, False),
    (256, 4, 3, False),
    (256, 4, (3,2), False),
    (512, 3, (3,2), False),
    (512, 3, 2, False),
]

decoder = [
    (512, 3, (1,2), True),
    (256, 4, (3,2), True),
    (256, 4, (3,2), True),
    (128, 4, (3,3), False),
    (128, 5, 5, False),
    (64, 4, 2, False),
    (64, 4, 2, False),
]
    
model = SpatialSuppression(encoder,decoder).model()
model.summary()
