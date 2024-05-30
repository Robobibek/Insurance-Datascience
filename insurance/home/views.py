from django.shortcuts import render
from django.shortcuts import HttpResponse
import joblib
import numpy as np

model = joblib.load('static/random_forest_regressor')
new_model = model
new_model.tree_ = np.array(model.tree_, dtype=[('left_child', '<i8'), ('right_child', '<i8'), ('feature', '<i8'), ('threshold', '<f8'), ('impurity', '<f8'), ('n_node_samples', '<i8'), ('weighted_n_node_samples', '<f8')])

# Save the new model
joblib.dump(new_model, 'static/random_forest_regressor_fixed')

# Create your views here.
def index(request):
    # return HttpResponse("this is the home page")
    return render(request,'index.html')

def about(request):
    return render(request,'about.html')

def contact(request):
    return render(request,'contact.html')

def prediction(request):
    if request.method =="POST":
        age = int(request.POST.get('age'))
        sex = int(request.POST.get('sex'))
        bmi = int(request.POST.get('bmi'))
        children = int(request.POST.get('children'))
        smoker = int(request.POST.get('smoker'))
        region = int(request.POST.get('region'))
        pred = round(model.predict([[age, sex, bmi, children, smoker, region]])[0])
        print(pred)

        output ={
            "output": pred
        }
        return render(request, 'prediction.html', output)
    else:
        return render(request,'prediction.html')