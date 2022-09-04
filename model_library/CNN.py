import tensorflow as tf
from tensorflow.keras import layers


class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.input_norm = layers.BatchNormalization()
        self.image_encoder_layers = [
            layers.Conv2D(filters=16, kernel_size=5, strides=1, activation='relu'),
            layers.MaxPooling2D(pool_size=2, strides=2, padding='valid', data_format=None),
            layers.BatchNormalization(),

            layers.Conv2D(filters=32, kernel_size=5, padding = 'SAME', strides=1, activation='relu'),
            layers.MaxPooling2D(pool_size=2, strides=2, padding='valid', data_format=None),
            layers.BatchNormalization(),

            layers.Conv2D(filters=64, kernel_size=3,  padding = 'SAME', strides=1, activation='relu'),
            layers.Conv2D(filters=64, kernel_size=3,  padding = 'SAME', strides=1, activation='relu'),
            layers.Conv2D(filters=64, kernel_size=3,  padding = 'SAME', strides=1, activation='relu'),
            layers.MaxPooling2D(pool_size=2, strides=2, padding='valid', data_format=None),

            layers.Conv2D(filters=128, kernel_size=3,  padding = 'SAME', strides=1, activation='relu'),
            layers.Conv2D(filters=128, kernel_size=2,  padding = 'SAME', strides=1, activation='relu'),
            layers.Conv2D(filters=128, kernel_size=3,  padding = 'SAME', strides=1, activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid', data_format=None),

            layers.Conv2D(filters=256, kernel_size=3,  padding = 'SAME', strides=1, activation='relu'),
            layers.Conv2D(filters=256, kernel_size=3,  padding = 'SAME', strides=1, activation='relu'),
            layers.Conv2D(filters=256, kernel_size=3,  padding = 'SAME', strides=1, activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid', data_format=None),        
            
            layers.Conv2D(filters=512, kernel_size=3,  padding = 'SAME', strides=1, activation='relu'),
            layers.Conv2D(filters=512, kernel_size=3,  padding = 'SAME', strides=1, activation='relu'),
            layers.Conv2D(filters=512, kernel_size=3,  padding = 'SAME', strides=1, activation='relu'),
            layers.Conv2D(filters=1024, kernel_size=2,  padding = 'SAME', strides=2, activation='relu'),
        ]
        
        self.output_layers = [
            layers.Dense(units=1000, activation='relu'),
            layers.Dropout(rate=0.2),
            layers.Dense(units=200, activation='relu'),
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
        total_image_counts, height, width, channels = encoded_images.shape
        encoded_image_sequences = tf.reshape(
            encoded_images, [batch_size, encode_length, height, width, channels]
        )

        flatten_feature = tf.reshape(encoded_image_sequences, [batch_size, -1])
        
        combine_feature = tf.concat([flatten_feature, feature], 1)

        # output block
        output = self.apply_list_of_layers(
            combine_feature, self.output_layers, training
        )
        return output
