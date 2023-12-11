# MarioRL2

MarioRL is an experiment of basic PPO improvements in the Super Mario Bros Enviroment. This REPO comes with basic training, eval code, as well as some pretrained models.

VIDEO:

[![Demo](https://i9.ytimg.com/vi/Xm55TjfA-bw/mqdefault.jpg?sqp=CPDQ2asG-oaymwEmCMACELQB8quKqQMa8AEB-AH-AYACkAGKAgwIABABGHIgRiguMA8=&rs=AOn4CLDViAyB3BsBlxRBpk3C_iMwZjOjOQ)](https://youtu.be/Xm55TjfA-bw?si=F2guDpNMy9GtOBdg&t=209)


### SETUP

Please note that the libraries we are using are old. You need to proper wheels for installation to begin, which in itself can be a hassle. Python 3.8.1 seems to be the easiest version that can build all the libraries. Depenendencies will are installed in the described notebooks.

### Training
Training can be done via Training.ipnyb. Each of the cells cooresponds to a certain type of model, with specified heuristics. To run this notebook, please make sure to download the correct version of Python (3.8.1), and run the necessary commands to install the needed libraries (these are included in both the Train.ipynb and the Evaluate.ipynb notebooks).

Within the file, you will see examples of the function, which use each of the heuristics. To use all heuristics, use the bottom-most cell to train model "FinalModelVC". This is a model which uses all the heuristic methods, and trains on all of the training set levels.

### Evaluation
Evaluation of the pretrained models can be done via the Evaluation.ipnyb notebook. This contains basic scripts for plotting results and recording videos. 
It currently is set up for pretrained models, but can be used for any model. Some pretrained models are available in this repository. To run this notebook, please use Python 3.8.1, and make sure to first run the cell to install all the necessary libraries (the cell is included in this notebook).

This contains the code to generate all the plots present in our report.
