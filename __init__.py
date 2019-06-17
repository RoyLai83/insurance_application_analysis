import flask
import dill
import pandas as pd
import lime
import lime.lime_tabular
from numpy import array
import base64

import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import os

app = flask.Flask(__name__)


with open('xgboost_ins.pkl', 'rb') as f:
    PREDICTOR = dill.load(f)
with open('lime_ins.pkl', 'rb') as f:
    explainer = dill.load(f)

@app.route('/', methods=['POST', 'GET'])
def test():
    '''Gets prediction using the HTML form'''
    np = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26', 'f27', 'f28', 'f29', 'f30', 'f31', 'f32', 'f33', 'f34', 'f35', 'f36', 'f37', 'f38', 'f39', 'f40', 'f41', 'f42', 'f43', 'f44', 'f45', 'f46', 'f47', 'f48', 'f49', 'f50', 'f51', 'f52', 'f53', 'f54', 'f55', 'f56', 'f57', 'f58', 'f59', 'f60', 'f61', 'f62', 'f63', 'f64', 'f65', 'f66', 'f67', 'f68', 'f69', 'f70', 'f71', 'f72', 'f73', 'f74', 'f75', 'f76', 'f77', 'f78', 'f79', 'f80', 'f81', 'f82', 'f83', 'f84', 'f85', 'f86', 'f87', 'f88', 'f89', 'f90', 'f91', 'f92', 'f93', 'f94', 'f95', 'f96', 'f97', 'f98', 'f99', 'f100', 'f101', 'f102', 'f103', 'f104', 'f105', 'f106', 'f107', 'f108', 'f109', 'f110', 'f111', 'f112', 'f113', 'f114', 'f115', 'f116', 'f117', 'f118', 'f119', 'f120', 'f121', 'f122', 'f123']
    med_value=[1.0, 16.0, 26.0, 0.2307, 2.0, 3.0, 1.0, 0.403, -0.7953659353322189, -2.7333304496957087, 9.0, 1.0, 0.0, 2.0, 0.15, 1.0, 2.0, 6.0, 3.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 2.0, 0.0001667, 1.0, 2.0, 2.0, 3.0, 0.2174, 0.0, 0.324, 0.0, 1.0986122886681098, 5.0, 2.0, 2.0, 1.0, 3.0, 2.0, 2.0, 2.0, 0.0, 3.0, 2.0, 3.0, 3.0, 0.0, 1.0, 3.0, 1.0, 1.0, 2.0, 1.0, 2.0, 3.0, 0.0, 1.0, 3.0, 3.0, 1.0, 3.0, 2.0, 3.0, 0.0, 3.0, 3.0, 1.0, 2.0, 2.0, 1.0, 3.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    if flask.request.method == 'POST':

       inputs = flask.request.form

       Medical_History_4 = inputs['Medical_History_4'][0]
       BMI = inputs['BMI'][0]
       Product_Info_4 = inputs['Product_Info_4'][0]
       Medical_History_15 = inputs['Medical_History_15'][0]
       InsuredInfo_6 = inputs['InsuredInfo_6'][0]
       Medical_History_23 = inputs['Medical_History_23'][0]
       Medical_Keyword_15 = inputs['Medical_Keyword_15'][0]
       Ins_Age = inputs['Ins_Age'][0]

       med_value[38]=float(Medical_History_4)
       med_value[8]=float(BMI)
       med_value[3]=float(Product_Info_4)
       med_value[49]=float(Medical_History_15)
       med_value[20]=float(InsuredInfo_6)
       med_value[57]=float(Medical_History_23)
       med_value[90]=float(Medical_Keyword_15)
       med_value[7]=float(Ins_Age)

       item = pd.DataFrame([med_value],columns=np)
       print(item)

       predicted_class = PREDICTOR.predict(item)
       predict_fn_xgboost = lambda x: PREDICTOR.predict_proba(x).astype(float)
       exp = explainer.explain_instance(item.values[0],predict_fn_xgboost, num_features=5)
       plt = exp.as_pyplot_figure()
       plt.tight_layout()
       plt.savefig('static/plot.png')
       exp.save_to_file('static/plot2.html')
       # full_filename = os.path.join(app.config['graph_folder'],'plot.png')
    #    class_1 = int(score[0,0] * 100)
    #    class_2 = int(score[0,1] * 100)
    #    class_3 = int(score[0,2] * 100)
    #    class_4 = int(score[0,3] * 100)
    #    class_5 = int(score[0,4] * 100)
    #    class_6 = int(score[0,5] * 100)
    #    class_7 = int(score[0,6] * 100)
    #    class_8 = int(score[0,7] * 100)
    #    score = PREDICTOR.predict_proba(item)

    else: #if not post then it is get, setting variables to 0
       Medical_History_4 = 0
       BMI = 0
       Product_Info_4 = 0
       Medical_History_15 = 0
       InsuredInfo_6 = 0
       Medical_History_23 = 0
       Medical_Keyword_15 = 0
       Ins_Age = 0
       predicted_class = 0
       item = pd.DataFrame([med_value],columns=np)
       predict_fn_xgboost = lambda x: PREDICTOR.predict_proba(x).astype(float)
       exp = explainer.explain_instance(item.values[0],predict_fn_xgboost, num_features=5)
       plt = exp.as_pyplot_figure()
       plt.tight_layout()
       

    return flask.render_template('dataentrypage.html',plot_url='./static/plot.png',Medical_History_4=Medical_History_4,BMI=BMI, Product_Info_4=Product_Info_4,
                                 Medical_History_15=Medical_History_15,InsuredInfo_6=InsuredInfo_6,Medical_History_23=Medical_History_23,
                                 predicted_class=predicted_class,exp=exp)



if __name__ == '__main__':
    app.run(debug=True)
