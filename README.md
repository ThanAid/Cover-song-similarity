# Cover-song-similarity

Cover-song-similarity is a Python project designed to calculate similarity scores between songs. It preprocesses musical data, trains a Siamese Convolutional Neural Network (CNN) to detect tonal similarities, and provides an API to inference the trained model. The system leverages the Da-TACOS dataset and employs Docker for seamless deployment.

Detailed analysis on this project can be found in this [report](Cover_Similarity_Aidinis.pdf)

---

## Dataset
The Da-TACOS benchmark subset is used for this project. It is designed for standardized evaluation, making it suitable for assessing cover detection systems. The dataset ensures reproducibility, allowing results to be compared with prior work. With 15,000 songs, it provides ample data for training, validation, and testing.

The dataset includes metadata such as title and artist, along with pre-extracted features like HPCP and Chroma. HPCP features are chosen for their robustness to key shifts and effectiveness in cover detection tasks. The dataset can be found [here](https://github.com/MTG/da-tacos)
.

# Data Exploration
A notebook with data a simple data exploration can be found [here](notebooks/data_exploration.ipynb)

## Installation

### Preprocessing Data

After downloading the `da-tacos_benchmark_subset_hpcp` set from this [link](https://drive.google.com/drive/folders/1GfFF_Kan_Qe69MF15i3-_LqE4wn3XNsb)
you can run the preprocess pipeline.
Preprocessing involves steps like normalization, adding Gaussian noise, and zero-padding to prepare the data for training.

Normalization scales HPCP feature values to a range of 0 to 1 using Min-Max normalization. Gaussian noise is added to improve model generalization and reduce overfitting by introducing slight perturbations. Zero-padding ensures uniform sequence length for all input data, preserving temporal structure.

Run the preprocessing pipeline with the following command, specifying the dataset directory and output directory. This command drops 50% of the data as set by the -drop parameter (less memory):
```bash
cd src
python -m preprocess -d /path/to/da-tacos_benchmark_subset_hpcp -o /path/to/data_folder -drop 0.5
```
This process will generate the `pairs.pickle` and the `pair_labels.pickle`.

```bash
$python -m modelling -h
Preprocess the data.

options:
  -h, --help            show this help message and exit
  -d DIRECTORY, --directory DIRECTORY
                        directory containing the data
  -o OUTPUT, --output OUTPUT
                        output directory to save the pairs
  -perc DIS_PERC, --dis_perc DIS_PERC
                        percentage of dissimilar pairs to create
  -drop DROP_PERC, --drop_perc DROP_PERC
                        percentage of data to drop
```

### Training the Model
Train the similarity model using the prepared data:

```bash
cd src
python -m modelling -X /path/to/pairs.pickle -y /path/to/pair_labels.pickle -sp model.h5
```

```bash
$python -m modelling -h 

Train the model.

options:
  -h, --help            show this help message and exit
  -X X_PATH, --X_path X_PATH
                        Path of the X data pickle file
  -y Y_PATH, --y_path Y_PATH
                        Path of the y data file
  -ts TEST_SIZE, --test_size TEST_SIZE
                        percentage of the test size
  -vs VAL_SIZE, --val_size VAL_SIZE
                        percentage of validation size
  -sp SAVE_PATH, --save_path SAVE_PATH
                        output directory to save the model
```


## Running with Docker
Spin up the Dockerized application by running the following command on the root directory of the project:
(depending on your version the command is either `docker compose` or `docker-compose`).
```bash
docker compose up
```

# Making Predictions via API
I have created some jsons containing a pair of tracks each (randomly). In the name of the json file the label is described (for example the following has label `0` meanining the two tracks are similar).

Test the model predictions using a sample JSON payload (run this on the root folder of the project):

```bash
curl -X POST -H "Content-Type: application/json" -d @inference_data/516_arrays_data_lbl_0.json http://0.0.0.0:8080/inference

curl -X POST -H "Content-Type: application/json" -d @inference_data/2080_arrays_data_lbl_1.json http://0.0.0.0:8080/inference
```

You can also get the summary of the model by running:

```bash
curl -X GET http://0.0.0.0:8080/model_summary
```

# Dowload model & run without Docker
You can also download the model by running:
```bash
pip install -r requirements.txt
python download_model.py output/path/model.h5
```

Add the model to a folder named `app` or in the `src/api/constants.py` script change the path accordingly.

run:
```bash
cd src
python -m uvicorn app:app --host 0.0.0.0 --port 8080 
```
# Running Tests
To run sample tests for the project:


Install pytest if not already installed:
```bash
pip install pytest
```

Run the test file:
```bash
pytest tests/test_create_pairs.py
```