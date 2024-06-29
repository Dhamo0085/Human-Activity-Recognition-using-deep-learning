# Deep Learning for Human Activity Recognition

Deep learning represents a significant advancement in human activity recognition (HAR), leveraging the capabilities of neural networks to achieve superior performance compared to traditional methods. This repository provides a demonstration of HAR using both TensorFlow and PyTorch frameworks.

## Prerequisites

- Python 3.x
- Numpy
- TensorFlow or PyTorch 1.0+

## Dataset

There are several public datasets available for human activity recognition. For this demonstration, we use the UCI HAR dataset as an example. You can find more information about this dataset in this [survey article](https://www.mdpi.com/1424-8220/18/11/3603).

We provide a preprocessed version of the dataset in `.npz` format for convenience, which you can download [here](link_to_your_preprocessed_dataset). However, we highly recommend downloading the original dataset to understand the preprocessing steps and explore the entire process firsthand.

### Dataset Details

- **Subjects**: 30
- **Activities**: 6
- **Frequency**: 50 Hz

## Usage

### PyTorch

1. Navigate to the `pytorch` folder.
2. Configure the data folder path in `config.py`.
3. Run `main_pytorch.py`.

### TensorFlow

1. Run `main_tensorflow.py`.
   
**Note**: TensorFlow updates are no longer actively maintained in this repository; we recommend using PyTorch for ongoing development.

## Network Structure

The core of our HAR model utilizes Convolutional Neural Networks (CNNs), which are well-suited for sequence and spatial data like sensor inputs.

### CNN Structure

Our CNN architecture consists of:

- Convolutional Layer + Pooling
- Convolutional Layer + Pooling
- Fully Connected (Dense) Layer
- Fully Connected (Dense) Layer
- Fully Connected (Dense) Layer (Output)

### Input Details

The dataset includes 9 channels of inputs: accelerometer (body, total) and gyroscope readings on x-y-z axes. Each input file represents a channel, and sequences are segmented into windows of 128 samples.

### Data Format

Inputs are reformatted to `[n_sample, 128, 9]`, where each window contains 9 channels, each with 128 samples. For TensorFlow, reshape inputs to `[n_sample, 9, 1, 128]`.

## Results

- **result.csv**: CSV file containing training and test accuracies across epochs.
- **plot.png**: Plot showing the training and test accuracies over epochs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- UCI Machine Learning Repository for providing the HAR dataset.
- TensorFlow and PyTorch communities for their powerful deep learning frameworks.

## Related Projects

Explore these related projects and resources:

- [Must-read papers about deep learning based human activity recognition](https://github.com/jindongwang/activityrecognition/blob/master/notes/deep.md)
- [LSTM-Human-Activity-Recognition](https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition)
- [Human-Activity-Recognition-using-CNN](https://github.com/aqibsaeed/Human-Activity-Recognition-using-CNN)
