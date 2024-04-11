from fastapi import FastAPI
import uvicorn
import pickle

app = FastAPI()


@app.get("/")
async def root():
    return {"text": "Breast Cancer Class Prediction"}

@app.get('/predict')
def predict(Clump_Thickness: int, Uniformity_of_Cell_Size: int, Uniformity_of_Cell_Shape: int, Marginal_Adhesion: int,
             Single_Epithelial_Cell_Size:int, Bare_Nuclei: int, Bland_Chromatin: int, Normal_Nucleoli: int, Mitosis: int):
    model = pickle.load(open('classification.pk1','rb'))
    makeprediction = model.predict([[Clump_Thickness, Uniformity_of_Cell_Size, Uniformity_of_Cell_Shape, Marginal_Adhesion, 
                                     Single_Epithelial_Cell_Size, Bare_Nuclei, Bland_Chromatin, Normal_Nucleoli, Mitosis]])
    output = makeprediction

    return {'The prediction grade is {}'.format(output)}

if __name__ == '__main__':
    uvicorn.run(app)
    