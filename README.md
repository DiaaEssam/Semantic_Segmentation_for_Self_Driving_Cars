# Semantic Segmentation for Self-Driving Cars

This repository contains code and notebooks for semantic segmentation in the context of self-driving cars.

## Table of Contents

- [Introduction](#introduction)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The project focuses on semantic segmentation, a computer vision task that aims to assign semantic labels to each pixel in an image. The goal is to develop accurate models for understanding the environment in self-driving cars.

You can install these dependencies using `pip`:

```
pip install tensorflow matplotlib flask pillow numpy opencv-python flasgger
```

## Usage

The project consists of two main components: `segment.py` and `API.py`.

### `segment.py`

This script provides functions for segmenting images using a trained model. It utilizes TensorFlow and Matplotlib libraries.

### `API.py`

This script sets up a Flask API for serving the segmentation functionality as a web service. It also uses the Flasgger library for generating API documentation.

To use the segmentation functionality, run the Flask API by executing the following command:

```
python API.py
```

## Examples

The `semantic-segmentation-for-self-driving-cars.ipynb` notebook demonstrates how to use the code for displaying images and masks in the dataset.

## Contributing

Contributions to this project are welcome! If you find any issues or have ideas for improvements, please submit a pull request.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
