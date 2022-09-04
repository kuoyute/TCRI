import tensorflow as tf
from tensorflow.keras import layers


class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.input_norm = layers.BatchNormalization()
        self.image_encoder_layers = [
            layers.Conv2D(filters=16, kernel_size=4, strides=2, activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=32, kernel_size=3, strides=2, activation='relu'),
            layers.BatchNormalization()
        ]
        
        self.cross_channel_attention =[
            layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=1, kernel_size=2, padding='same', strides=1),
            #layers.Softmax()
        ]
        
        self.rnn_block = layers.ConvLSTM2D(
            filters=64, kernel_size=4, dropout=0.0,
            recurrent_dropout=0.0, return_sequences=False
        )
        self.rnn_output_encoder = layers.Conv2D(filters=64, kernel_size=1, strides=1, activation='relu')
        self.output_layers = [
            layers.Dense(units=128, activation='relu'),
            layers.Dropout(rate=0.2),
            layers.Dense(units=1),
        ]

    def apply_list_of_layers(self, input, list_of_layers, training):
        x = input
        for layer in list_of_layers:
            x = layer(x, training=training)
        return x
    
    def auxiliary_feature(self, feature):
 #   feature: ['starting_land_dis', 'ending_land_dis', 'translation_speed', 'starting_intensity', 'starting_lat', 'ending_lat']
        land_distance = feature[:, 0:2]
        translation_speed = feature[:, 2:6]
        return tf.concat([land_distance, translation_speed], 1)

    def call(self, image_sequences, feature, training):
        batch_size, encode_length, height, width, channels = image_sequences.shape

        # image_encoder block
        images = tf.reshape(
            image_sequences, [batch_size*encode_length, height, width, channels]
        )
        normalized_images = self.input_norm(images, training=training)
        encoded_images = self.apply_list_of_layers(
            normalized_images, self.image_encoder_layers, training
        )
        
        # cross_channel mask
        cca_mask = self.apply_list_of_layers(
            encoded_images, self.cross_channel_attention, training
        )
        cca_images = cca_mask * encoded_images
        
        total_image_counts, height, width, channels = cca_images.shape
        encoded_image_sequences = tf.reshape(
            cca_images, [batch_size, encode_length, height, width, channels]
        )

        # rnn block
        feature_sequences = self.rnn_block(encoded_image_sequences, training=training)       

        # rnn_output_encoder block
        compressed_features = self.rnn_output_encoder(
            feature_sequences, training=training
        )
        flatten_feature = tf.reshape(compressed_features, [batch_size, -1])
        
   #     auxiliary_feature = self.auxiliary_feature(feature)
        combine_feature = tf.concat([flatten_feature, feature], 1)

        # output block
        output = self.apply_list_of_layers(
            combine_feature, self.output_layers, training
        )
        return output
