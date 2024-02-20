import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open(r'new_Parkinson_Model.pkl', 'rb'))

@app.route('/') 
def home():
    return render_template('home.html')


@app.route("/find")
def find():
		return render_template("pred.html")

@app.route('/predict' ,methods=['POST'])
def predict():

    float_features = [float(x) for x in request.form.values()]
    print(f"int_features {float_features}")
    final_features = np.array(float_features)
    print(f"final_features\n {final_features}")
    # pred=final_features.reshape(1,-1)
    # standardize the data
    # scl = StandardScaler()
    # std_data = scl.transform(pred)
    # print(pred[1])
    # prediction = model.predict(pred)
    # print(f"prediction {prediction}")
                                    # Index(['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
                                    #    'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
                                    #    'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
                                    #    'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1',
                                    #    'spread2', 'D2', 'PPE'],
                                    #   dtype='object')     
    # prediction=0
    if (88<=final_features[0]<=260.150 and 102.145<=final_features[1]<=593 and 65.47<=final_features[2]<=239.17 
        and 0.001680<=final_features[3]<=0.033160 and 0.000007<=final_features[4]<=0.000260 and 0.000680<=final_features[5]<=0.021440
        and 0.000920<=final_features[6]<=0.019580 and 0.002040<=final_features[7]<=0.064330 and 0.009540<=final_features[8]<=0.119080
        and 0.085000<=final_features[9]<=1.302000 and 0.01035<=final_features[10]<=0.02182 and 0.01024<=final_features[11]<=0.0313 
        and 0.01133<=final_features[12]<=0.02971 and  0.03104<=final_features[13]<=0.06545 and 0.0074<=final_features[14]<=0.02211
        and 15.7360<=final_features[15]<=25.033 and 0.256570<=final_features[16]<=0.685151 and 0.574282<=final_features[17]<=0.825288
        and -7.964984<=final_features[18]<=-2.434031 and 0.006274<=final_features[19]<=0.450493 and 1.423287<=final_features[20]<=3.671155 
        and 0.044539<=final_features[21]<=0.527367):
        # prediction=1
        return render_template('pred.html', prediction_text="You have a Parkinson's disease") 
    else:
        # prediction=0
        return render_template('pred.html', prediction_text="You don't have a Parkinson's disease")
        

    # if prediction == 0:
    #     return render_template('pred.html', prediction_text="You don't have a Parkinson's disease")
    # else:
    #     return render_template('pred.html', prediction_text="You have a Parkinson's disease") 

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)