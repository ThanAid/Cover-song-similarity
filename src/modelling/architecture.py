import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np

def create_base_network(input_shape):
    """Create the base network for the Siamese network"""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, 3, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        layers.Conv1D(128, 3, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))
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

    # # Compute absolute difference between embeddings
    # distance = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([processed_a, processed_b])
    
    # # Output layer
    # outputs = layers.Dense(1, activation='sigmoid')(distance)
    outputs = CosineSimilarityLayer()([processed_a, processed_b])

    # Model
    siamese_model = tf.keras.Model(inputs=[input_a, input_b], outputs=outputs)
    return siamese_model

def contrastive_loss(y_true, y_pred):
    # TODO maybe something better
    margin = 1
    # For similar pairs (y_true = 0), minimize y_pred
    # For dissimilar pairs (y_true = 1), maximize y_pred to be greater than margin
    loss_similar = (1 - y_true) * tf.square(y_pred)
    loss_dissimilar = y_true * tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(loss_similar + loss_dissimilar)

# def contrastive_loss(y_true, y_pred):
#     """Contrastive loss function"""
#     margin = 1.0
#     return tf.reduce_mean(y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0)))

# def custom_accuracy(y_true, y_pred):
#     """Custom accuracy metric for Siamese network"""
#     threshold = 0.5
#     predicted_labels = (y_pred < threshold).astype(int).flatten()
#     return np.mean(predicted_labels == y_true)

def custom_accuracy(y_true, y_pred):
    """Custom accuracy metric for Siamese network"""
    threshold = 0.5
    predicted_labels = tf.cast(y_pred > threshold, tf.int64)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_true, predicted_labels), tf.float32))
    return accuracy

if __name__ == "__main__":
        # Compile the model
    input_shape = (128, 12)  # Replace with the shape of your NumPy array
    siamese_model = create_siamese_network(input_shape)
    siamese_model.compile(optimizer='adam', loss=contrastive_loss, metrics=["accuracy"])

    # Example training data
    # pairs: list of tuple pairs of numpy arrays (e.g., [(x1, x2), (x3, x4)])
    # labels: list of 0/1 labels indicating similarity
    pairs = [(np.random.rand(128, 12), np.random.rand(128, 12)) for _ in range(1000)]
    labels = np.random.randint(0, 2, size=(1000,))
    print(labels[:10])

    # Prepare inputs for training
    input_a = np.array([pair[0] for pair in pairs])
    input_b = np.array([pair[1] for pair in pairs])

    # Train the model
    siamese_model.fit([input_a, input_b], labels, batch_size=32, epochs=2)

    # Save the model
    siamese_model.save("siamese_model.h5")

    # Load the model for inference
    loaded_model = tf.keras.models.load_model("siamese_model.h5", custom_objects={'contrastive_loss': contrastive_loss, 'custom_accuracy': custom_accuracy, 'CosineSimilarityLayer': CosineSimilarityLayer})

    # Prepare input data for inference
    test_pairs = [(np.random.rand(128, 12), np.random.rand(128, 12)) for _ in range(10)]
    test_input_a = np.array([pair[0] for pair in test_pairs])
    test_input_b = np.array([pair[1] for pair in test_pairs])
    test_labels = np.random.randint(0, 2, size=(10,))

    # Perform inference
    predictions = loaded_model.predict([test_input_a, test_input_b])

    # Inspect the outputs
    for i, prediction in enumerate(predictions):
        print(f"Pair {i}: Similarity score = {prediction[0]}, true label: {test_labels[i]}, predicted label: {0 if prediction[0] < 0.5 else 0}")

    # Calculate accuracy on the test set
    threshold = 0.5
    predicted_labels = (predictions > threshold).astype(int).flatten()
    accuracy = np.mean(predicted_labels == test_labels)
    print(f"Test set accuracy: {accuracy:.4f}")