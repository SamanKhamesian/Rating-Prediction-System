# Rating-Prediction-System

### Abstract
This project implemented a prediction of rating of items using latent factor model

##

#### To use this work on your researches or projects you need:
* Python 3.7.0
* Python packages:
	* Numpy
##

#### To install Python:
_First, check if you already have it installed or not_.
~~~~
python3 --version
~~~~
_If you don't have python 3 in your computer you can use the code below_:
~~~~
sudo apt-get update
sudo apt-get install python3
~~~~
##

#### To install packages via pip install:
~~~~
sudo pip3 install numpy
~~~~
_If you haven't installed pip, you can use the codes below in your terminal_:
~~~~
sudo apt-get update
sudo apt install python3-pip
~~~~
_You should check and update your pip_:
~~~~
pip3 install --upgrade pip
~~~~
##

#### Dataset
* raw_data.txt : Each rows should contain the userID, the itemID and the predicted rating respectively.
* train_user_item_score.txt : 70% of raw_data.txt chose randomly for train data.
* validation_user_item_score.txt : the remain of raw_data after creation train data made validation data.