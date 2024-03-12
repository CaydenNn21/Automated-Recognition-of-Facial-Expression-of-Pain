# Introduction

- This is the academic project with a title Automated Recognition of Facial Expression of Pain Using Machine Learning developement files.
- This project files contains two main folder:
  - First folder: '\Deploy', this is the folder that store all the deployment files
  - Second Folder: '\Module', this is the folder where it contains two folders which are '\Execute' and '\Test' respectively
- In '\Module\Execute' folder contains all the files used to do model training and preprocess the dataset
- In '\Module\Test' folder contains all the experimental file to create some specific output for investigation
- 'Module\Execute\model-training-pain-detector.ipynb' this is the model training jupyter notebook that run on Kaggle with all the outputs

# Instruction to run the application on local device

1. Under the folder namely "Deploy" contains the deployment file of the application
2. First, install all required libraries from the 'requirements.txt'
3. After installing the file, run the app3.py python script to run the program
4. In the command prompt, will see a local address that used to host the Flask application
5. Click on the '127.0.0.1:5000' or copy this to the browser to run the program

## Take Note

Please allow the server to run at the backend for around 1-2 mins before prompting the result in the interface.
The waiting timeframe will depends on the available hardware in localhost, the program will run faster with GPU available

