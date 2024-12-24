"""Training pipeline."""
import time
from loguru import logger
from modelling.architecture import create_siamese_network, contrastive_loss, CosineSimilarityLayer
from modelling.train_utils import DataGenerator, load_split_data
import argparse
import tensorflow as tf
import numpy as np

def arg_parser(args=None):
    """Argument parser."""
    parser = argparse.ArgumentParser(description="Train the model.")

    parser.add_argument("-X", "--X_path", type=str, action="store",
                        required=True, help="Path of the X data pickle file")
    parser.add_argument("-y", "--y_path", type=str, action="store",
                        required=True, help="Path of the y data file")
    parser.add_argument("-ts", "--test_size", type=float,
                        default=0.15, required=False, help="percentage of the test size")
    parser.add_argument("-vs", "--val_size", type=float,
                        default=0.25, required=False, help="percentage of validation size")
    parser.add_argument("-sp", "--save_path", type=str, action="store",
                        required=False, default="siamese_model.h5" ,help="output directory to save the model")
    return parser.parse_args(args)


def inference_model(X, y, model) -> tuple:
    """Inference model and calculate accuracy"""
    train_gen = DataGenerator(X, y, batch_size=32)

    predictions = model.predict(train_gen)

    threshold = 0.5
    predicted_labels = (predictions > threshold).astype(int).flatten()
    accuracy = np.mean(predicted_labels == y)
    logger.info(f"Train set accuracy: {accuracy:.4f}")
    return predictions, predicted_labels

def main(X_path: str, y_path: str, test_size: float,
          val_size: float, save_path: str) -> None:
    """Main function to run the training pipeline."""
    logger.info("Loading data..")

    X_train, X_val, X_test, y_train, y_val, y_test = load_split_data(X_path,
                    y_path, test_size=test_size, validation_size=val_size)
    
    logger.info(f"""
    Data loaded successfully.
    Train size: {len(X_train)}
    0's in train: {y_train.count(0)}
    Validation size: {len(X_val)}
    0's in val: {y_val.count(0)}
    Test size: {len(X_test)}
    0's in test: {y_test.count(0)}
    {"-"*100}
    """)

    input_shape = X_train[0][0].shape
    logger.info(f"Input shape: {input_shape}")

    # Create data generators
    train_gen = DataGenerator(X_train, y_train, batch_size=32)
    val_gen = DataGenerator(X_val, y_val, batch_size=32)

    siamese_model = create_siamese_network(input_shape)
    siamese_model.compile(optimizer='adam', loss=contrastive_loss, metrics=["accuracy"])


    logger.info("Fitting the model..")
    siamese_model.fit(train_gen, validation_data=val_gen, epochs=6, verbose=1)

    # Save the model
    logger.info("Saving the model..")
    siamese_model.save(save_path)

    logger.success("Training completed successfully!")


    logger.info("Loading the model for inference..")

    model = tf.keras.models.load_model(save_path,
        custom_objects={'contrastive_loss': contrastive_loss,
                         'CosineSimilarityLayer': CosineSimilarityLayer})
    
    # Print model architecture
    logger.info("Model architecture:")
    model.summary()

    # Perform inferencing on the test set, train set and validation set
    logger.info("Performing inferencing on the train set..")
    inference_model(X_train, y_train, model)

    logger.info("Performing inferencing on the validation set..")
    inference_model(X_val, y_val, model)

    logger.info("Performing inferencing the test set..")
    inference_model(X_test, y_test, model)

    return None

if __name__ == "__main__":
    start = time.time()
    logger.info("Starting up..")

    args = vars(arg_parser())
    main(**args)
    
    time_elapsed = time.time() - start
    logger.info(f"Time elapsed: {time_elapsed:.2f} seconds.")
    
    