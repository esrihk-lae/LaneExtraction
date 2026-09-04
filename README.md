This is a custom internal-use version of the Lane-Level Street Map Extraction project.

Not intended for external use.




# Lane-Level Street Map Extraction from Aerial Imagery
## Abstract
Digital maps with lane-level details are the foundation of many applications. However, creating and maintaining digital maps especially maps with lane-level details, are labor-intensive and expensive. In this work, we propose a mapping pipeline to extract lane-level street maps from aerial imagery automatically. Our mapping pipeline first extracts lanes at non-intersection areas, then it enumerates all the possible turning lanes at intersections, validates the connectivity of them, and extracts the valid turning lanes to complete the map. We evaluate the accuracy of our mapping pipeline on a dataset consisting of four U.S. cities, demonstrating the effectiveness of our proposed mapping pipeline and the potential of scalable mapping solutions based on aerial imagery.

## Environment

This code has been re-adapted and tuned to run on the following:

- Ubuntu Linux 24.04 (not wsl)
- Python versions: Python 3.5.2 to 3.8
- Tensorflow version: 1.15.0 (gpu strongly preferred)
- CUDA version: 10.0
- CUDNN version: 7
- NVIDIA driver version: 418.165.02

See setup-env.sh for environment setup details.

For small scale work, refer to setup-env.sh and use the 'tensorflow:20.12-tf1-py3' container available from nvcr.io


## View the dataset and annotate new images

Please check the instructions in [code/hdmapeditor](code/hdmapeditor).

## Create training data

We have the raw dataset in the [dataset](dataset) folder. To train the models, we have to first create the necessary data, e.g., the lane segmentation.

```bash
cd code
python3 create_training_data.py
```

## Training

Train the lane-and-direction extraction model:

```bash
cd code/laneAndDirectionExtraction
python3 train.py resnet34v3
```

Train the turning lane validation model:

```bash
cd code/turningLaneValidation
python3 train.py
```

Train the turning lane extraction model:

```bash
cd code/turningLaneExtraction
python3 train.py
```


### Testing

```bash
cd code/laneAndDirectionExtraction
./run_inference.sh
```


```bash
cd code/turningLaneExtraction
./run_inference.sh
```



### Random notes
[https://medium.com/@milad.4274/conclusion-graph-simplification-in-openstreetmap-is-a-crucial-preprocessing-step-for-network-analys-815e473b0d9f](OpenStreetMap Graph Simplification: A Deep Dive)