<h1 align='center'> Disease Detection in Leaves </h1> 
<h2 align='center'> Pytorch Lightning + W&B Tracking </h2>

<p align='center'><img src="assets/title.png"></p>
<p align='center'>
<img alt="Python" src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/>

<img alt="NumPy" src="https://img.shields.io/badge/numpy%20-%23013243.svg?&style=for-the-badge&logo=numpy&logoColor=white" />

<img alt="Pandas" src="https://img.shields.io/badge/pandas%20-%23150458.svg?&style=for-the-badge&logo=pandas&logoColor=white" />

<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white" />

<img alt="PyTorch Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white" />

</p>

## Introduction 

This repository contains the code for disease classification in a agro-crop. Data augmentation is performed using Albumentations library is training is performed using PyTorch Lightning framework. 

## Training

### 1. Installing the dependencies

To run the code in this repository, few frameworks need to be installed in your machine. 
Make sure you have enough space and stable internet connection.

Run the below command for installing the required dependencies.

```shell
$ pip install -r requirements.txt
```
### 2. Get the data

Data can retreived from the public datasets like Plant Village Dataset. For more information [visit here](https://www.tensorflow.org/datasets/catalog/plant_village)

Now, take the downloaded `.zip` file and extract it into the new folder `input/`.

Take care that the `input/` folder is at same directory level as `train.py` file.

### 3. Training the model

If you have the above steps right then, running the train.py should not produce any errors. 
To run the code, open the terminal and change the directory level same as `train.py` file. 
Now run the `train.py` file.

```shell
$ python train.py
```
You should start seeing the progress bar, on few seconds at the beginning of training.
If you have any problem, feel free to open a Issue. Will be happy to help.




