"""Architecture of SNN Model."""
import tensorflow as tf
from tensorflow.keras import layers, models

def create_base_network(input_shape):
    """Create the base network for the Siamese network"""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Dropout(0.4),
        layers.Conv1D(128, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu')
    ])
    return model

class CosineSimilarityLayer(layers.Layer):
    """
    Custom layer to compute cosine similarity.
    
    This layer computes the cosine similarity between two vectors.
    ranges from 0 (similar) to 1 (not similar).
    """
    def call(self, inputs):
        x1, x2 = inputs
        # Normalize the vectors
        x1 = tf.math.l2_normalize(x1, axis=-1)
        x2 = tf.math.l2_normalize(x2, axis=-1)

        cosine_similarity = tf.reduce_sum(x1 * x2, axis=-1, keepdims=True)
        # TODO: is that the correct? 0 is similar, 1 is not similar?

        return 1 - cosine_similarity

def create_siamese_network(input_shape):
    """Create the Siamese network"""
    base_network = create_base_network(input_shape)

    input_a = tf.keras.Input(shape=input_shape)
    input_b = tf.keras.Input(shape=input_shape)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    outputs = CosineSimilarityLayer()([processed_a, processed_b])

    siamese_model = tf.keras.Model(inputs=[input_a, input_b], outputs=outputs)
    return siamese_model

def contrastive_loss(y_true, y_pred):
    margin = 1
    # For similar pairs (y_true = 0), minimize y_pred
    # For dissimilar pairs (y_true = 1), maximize y_pred to be greater than margin
    loss_similar = (1 - y_true) * tf.square(y_pred)
    loss_dissimilar = y_true * tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(loss_similar + loss_dissimilar)

def custom_accuracy(y_true, y_pred):
    """Custom accuracy metric for Siamese network"""
    threshold = 0.5
    predicted_labels = tf.cast(y_pred > threshold, tf.int64)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_true, predicted_labels), tf.float32))
    return accuracy