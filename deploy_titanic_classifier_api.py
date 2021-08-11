from fastapi import FastAPI
from fastapi.responses import JSONResponse
from deploy_titanic_classifier_object import Titanic_Classifier
from pydantic import BaseModel

#create the application
app = FastAPI(
    title = "Titanic Classifier API",
    version = 1.0,
    description = "Simple API to make predict survivor of Titanic."
)

#creating the classifier
classifier = Titanic_Classifier("deploy_titanic_classifier.pkl")

#Model
class Titanic(BaseModel):
    Pclass:int
    Age:float
    SibSp:int
    Parch:int
    Fare:float
    Sex_female:int
    Sex_male:int

@app.post("/",tags = ["titanic_classifier"])
def get_prediction(features:Titanic):
    survivors_pred = classifier.make_prediction(features.dict())
    return JSONResponse({"survivor":survivors_pred})