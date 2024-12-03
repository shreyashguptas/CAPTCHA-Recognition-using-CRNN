# CAPTCHA Prediction using Convolutional Recurrent Neural Network

This project focuses on training a Convolutional Recurrent Neural Network (CRNN) model to accurately predict 5-letter and 5-number CAPTCHA images. The goal is to develop a robust model that can reliably identify the text content within CAPTCHA images, which are commonly used as a security measure to distinguish between human users and automated bots.

## Dataset

The dataset used in this project consists of CAPTCHA images, which were downloaded from the [Kaggle dataset](https://www.kaggle.com/datasets/fournierp/captcha-version-2-images). The dataset is divided into the following proportions:

- Training set: 75% (approximately 803 images)
- Validation set: 12.5% (approximately 135 images)
- Test set: 12.5% (approximately 135 images)

## Project Structure

The file structure of the project is as follows:

```
.
├── data
│   └── CAPTCHA Images
│       ├── test
│       ├── train
│       └── val
├── dataset.py
├── model.py
├── output
│   ├── log.txt
│   ├── loss.png
│   └── weight.pth
├── predict.py
├── README.md
├── split_train_val_test.py
├── train.py
└── utils.py
```

## Getting Started

To run this project after cloning, follow these steps:

1. Update the folder location of the CAPTCHA images in the following files:
   - `dataset.py`: Line 56
   - `predict.py`: Lines 23 and 27
   - `split_train_val_test.py`: Line 38
   - `train.py`: Lines 80 and 67

2. Run the `train.py` file to train the model:
   ```
   python train.py
   ```

3. Run the `predict.py` file to get the final prediction accuracy:
   ```
   python predict.py
   ```

## Model Performance

The current prediction accuracy of the model is 92.5%.

## Training Visualization

Training and Validation Loss

The image above shows the training and validation loss over the course of model training. The blue line represents the training loss, while the orange line represents the validation loss. Both losses decrease rapidly in the initial epochs and then stabilize, indicating that the model is learning effectively without overfitting.

## Docker Container

When creating a Docker container for this project, make sure to use the 'all-gpu' tag to utilize GPUs if available. For example:

```
docker run --gpus all -it your_image_name:all-gpu
```

If you're running on a Mac or need more information on using GPUs with Docker, please consult the [Docker documentation](https://docs.docker.com/config/containers/resource_constraints/#gpu) for specific instructions.

## Personal Notes

1. This project is currently running in a Docker container on a server with an NVIDIA GTX 3060 Ti GPU.

2. It's fascinating to observe how methods originally designed to distinguish between robots and humans are becoming less effective. As we make progress in machine learning and hardware computation improves, training these models becomes easier and faster.

<video width="640" height="360" controls>
  <source src="/home/dev/dev_work_shrey/playing_around/pytorch_video.mp4" type="video/quicktime">
  Your browser does not support the video tag.
</video>

3. As a developer aspiring to build more in the AI/ML space, it's remarkable to realize that with readily available computing power and a few hours of work, we can build a model that achieves 92.5% accuracy in predicting CAPTCHAs – a task once thought to be an effective way to distinguish between humans and robots.

## Contributing

Contributions to this project are welcome. Please feel free to submit a Pull Request.

Citations:
[1] https://pplx-res.cloudinary.com/image/upload/v1719281556/user_uploads/afidsgvnv/loss.jpg
