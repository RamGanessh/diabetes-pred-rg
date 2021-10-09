from django.shortcuts import render
import pickle
import numpy as np
def home(request):
    return render(request, 'home.html')
def predict(request):
    return render(request, 'predict.html')
def result(request):
    knn = pickle.load(open('diabetes_knn.pkl', 'rb'))
    dtree = pickle.load(open('diabetes_dtree.pkl', 'rb'))
    lr = pickle.load(open('diabetes_lr.pkl', 'rb'))
    Pregnancy=float(request.GET["Pregnencies"])
    Glucose=float(request.GET["Glucose"])
    BP=float(request.GET["BP"])
    skin_Thickness=float(request.GET["ST"])
    Insulin=float(request.GET["Insulin"])
    BMI=float(request.GET["BMI"])
    pedigree_function=float(request.GET["PF"])
    Age=int(request.GET["Age"])
    inputs=[Pregnancy,Glucose,BP,skin_Thickness,Insulin,BMI,pedigree_function,Age]
    inputs=np.array([inputs])
    ans=[]
    ans.append(knn.predict(inputs))
    ans.append(lr.predict(inputs))
    ans.append(dtree.predict(inputs))
    result=''
    if ans.count(0)<=1:
        result='RESULT IS POSITIVE !!!'
    else:
        result='THE RESULT IS NEGATIVE !!!'
        
    return render(request, 'predict.html', {'ans':result})