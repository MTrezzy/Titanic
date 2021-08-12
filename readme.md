Titanic - Machine Learning from Disaster
==============================

This is a simple solution for the Titanic ML competition. The goal is to use machine learning to create a model that predicts which passengers survived the Titanic shipwreck.

My model use the Random Forest method to classify which passengers will survived the shipwreck. This simple solution get me a precision of 78.9% and rank in the top 4%.


## Online model
You can directly try the model at this address:
[Titanic model](https://mtrezzy-titanic.herokuapp.com/docs#/titanic_classifier/get_prediction__post)

## Local Installation
In order to run the notebooks, one must intall all the requirements.

* install python and pip on your machine.
  - on [windows](https://www.youtube.com/watch?v=otmWEEFysms)
  - on [linux](https://www.youtube.com/watch?v=Yg9AkozItTU)
  - on [mac](https://www.youtube.com/watch?v=XUaJ8OctxdM)

* Install the requirered python libraries. The list of required libraries are available in the file requirements.txt. It is preferable to use a [virtual environment](https://python-guide-pt-br.readthedocs.io/fr/latest/dev/virtualenvs.html).

```bash
# create virtual environment
virtualenv env -p python3
. env/bin/activate
pip install -r requirements.txt
```
