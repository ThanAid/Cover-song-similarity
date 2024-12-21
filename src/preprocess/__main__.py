"""Main script to run the preprocess pipeline."""
from preprocess.dictify import Dictify
from preprocess.prepro_pipe import PreproPipeline
from loguru import logger
import argparse
import time
import gc

def arg_parser(args=None):
    """Argument parser."""
    parser = argparse.ArgumentParser(description="Preprocess the data.")

    parser.add_argument("-d", "--directory", type=str, action="store",
                        required=True, help="directory containing the data")
    parser.add_argument("-o", "--output", type=str, action="store",
                        required=True, help="output directory to save the pairs")
    parser.add_argument("-perc", "--dis_perc", type=float,
                        default=0.4, required=False, help="percentage of dissimilar pairs to create")
    parser.add_argument("-drop", "--drop_perc", type=float,
                        default=0.0, required=False, help="percentage of data to drop")
    return parser.parse_args(args)

def main(directory: str, save_path: str, dissimilar_percentage: float, drop_percentage: float=0.0) -> None:
    """Main function to run the preprocess pipeline."""
    logger.info("Reading the data into a dictionary..")
    dictify = Dictify(directory=directory)
    data = dictify.get_data()
    max_length = dictify.get_max_length()

    # clear up memory
    dictify = None
    gc.collect()

    # Keep percentage of data by deleting keys
    if drop_percentage > 0.0:
        logger.info(f"Keeping {((1-drop_percentage)*100):.2f} % of the data..")
        data = {k: v for i, (k, v) in enumerate(data.items()) if i < int((1-drop_percentage)*len(data))}

    logger.info("Preprocessing the data..")
    prepro_pipe = PreproPipeline(data=data, dissimilar_percentage=dissimilar_percentage,
                                  max_length=max_length,save_path=save_path)
    prepro_pipe.run_pipeline()
    return None

if __name__ == "__main__":
    start = time.time()
    logger.info("Starting up..")

    args = arg_parser()
    main(args.directory, args.output, args.dis_perc, args.drop_perc)

    time_elapsed = time.time() - start
    logger.info(f"Time elapsed: {time_elapsed:.2f} seconds.")