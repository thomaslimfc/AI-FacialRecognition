# Image Processing & Computer Vision: Facial Recognition
- Assignment Title: Facial Detail Estimation
- Grade: A
- Bachelor of Software Engineering (RSW) Year2 Sem1 [TARUMT Penang]
- Course: BACS2003 Artificial Intelligence



## Description
This project aims to view the accuracy of three different facial recognition models, ResNet-50, VGG-16, and DeepFace, in estimating age and gender of faces detected from a video feed, and to measure the performance of the models by using metrics such as precision, recall, F1 score and MAE.

	Each model underwent transfer learning using a dataset from UTKFace to fine tune the data for the specific need of this project. Further details are elaborated in the report.

## Usage
Please install all the necessary libraries before running the program to ensure smooth operation.

	The program to run the live-video UI interface can be run from Main.py in the App folder. The main file will launch a UI that will allow you to select a model to use for real-time facial detail estimation. The detected faces will have a bounding box drawn around the face with the details printed below the bottom border of the bounding box.

	The program scripts to run the MAE test and Classification Report can be found as test.py and report.py under each of the group member's folders. The test file may take at least 3 minutes to run, as it will scan through the test dataset before outputting a result. The report file will print the classification report with the metrics Precision, Recall, F1 Score, Support, and macro and weighted averages.

# Collaborators:
- [Chin Gian Terng](mailto:chingt-pm23@student.tarc.edu.my)
- [Goh Wei Zhun](mailto:gohwz-pm23@student.tarc.edu.my)
- [Thomas Lim](mailto:limfc-pp21@student.tarc.edu.my)
<br>Their dedication and expertise made this repository possible, ensuring compatibility and functionality across various components.
