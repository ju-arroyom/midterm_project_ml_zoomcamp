import pickle
from flask import Flask, request, jsonify


def load_artifacts(path):
    with open(path, 'rb') as f_in:
        artifact = pickle.load(f_in)
    return artifact


def predict_client(dv, model, client):
    X = dv.transform([client])  
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]


dv, model = load_artifacts("./artifacts/model.bin")
app = Flask('midterm_project')


@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()
    prediction = predict_client(dv, model, client)

    decision = prediction >= 0.5
    
    result = {
        'probability': float(prediction),
        'decision': bool(decision),
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8787)