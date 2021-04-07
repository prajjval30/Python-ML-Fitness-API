from flask import Flask, jsonify, abort, request
from werkzeug.exceptions import HTTPException
from flask_cors import CORS
from sklearn import tree

app = Flask(__name__)
CORS(app)
app.config["DEBUG"] = True

@app.errorhandler(Exception)
def handle_error(e):
    code = 500
    if isinstance(e, HTTPException):
        code = e.code
    return jsonify(error=str(e)), code

@app.route('/api/get-message', methods=['GET'])
def getMessage():
    data = {'message': 'Hello World!'}
    return jsonify(data)

@app.route('/api/predict-fitness', methods = ['POST'])
def predict_fitness():
    if not request.json or not 'userHeight' in request.json:
        abort(400)

    # Data Cleaning, remvoing dot from userHeight value

    userHeight = request.json.get('userHeight', "")
    userHeight = int(str(userHeight).replace('.', ''))
    userWeight = int(request.json.get('userWeight', ""))

    if userHeight == 51: # Just a fix because python is removing trailing zeros while coverting to str or int
       userHeight = 510

    # Set up training data

    # Expected User Inputs to classifier

    # Example, For a person with 5 feet height, the expected weight is 43 to 53 kg
    # So for this we mention in features like : features = [[5, 43], [5, 99],....]] 
    # which means for 5 feet height, user can enter values from 43 to 99 
    # but the expected weight will be 4353 as mentioned in labels like: labels = [4353, 4353,.....]

    features = [[5, 43], [5, 99], [51, 45], [51, 99], [52, 48], [52, 99], [53, 50], [53, 99], 
                [5.4, 53], [54, 99], [55, 55], [55, 99], [56, 58], [56, 99], [57, 60], [57, 99], 
                [58, 63], [58, 99], [59, 65], [59, 99], [510, 67], [510, 99], [511, 70], [511, 99], 
                [6, 72], [6, 99], [6, 72], [6, 99]]

    # Expected output values based on user inputs or expected weight ranges based on person height

    labels = [4353, 4353, 4555, 4555, 4859, 4859, 5061, 5061, 5365, 5365 ,5558 ,5558, 5870,
            5870, 6074,6074 , 6376, 6376 ,6580 , 6580 ,6783 , 6783, 7085, 7085, 7289, 7289, 7289, 7289]
    
    # Train classifier

    classifier = tree.DecisionTreeClassifier()  # Decision tree classifier is used
    classifier = classifier.fit(features, labels)  # Find common patterns in training data

    # Make predictions using the trained model

    expectedWeight = classifier.predict([[userHeight,userWeight]])

    # Get first two numbers from expected Weight

    expectedWeight = int(expectedWeight)
    fromEpectedWeight = int(str(expectedWeight)[:2])

    # Get last two numbers from expected Weight

    toExpectedWeight = int(str(expectedWeight)[2:4])

    # Check if weight is in between the range of expected weight

    is_Weight_In_between = userWeight >= fromEpectedWeight and userWeight <= toExpectedWeight

    if is_Weight_In_between:
     message = f'Congratulations!, Your expected weight is in between {fromEpectedWeight} kg and {toExpectedWeight} kg.'
    else:
     message = f'Your expected weight should be in between {fromEpectedWeight} kg and {toExpectedWeight} kg.'


    fitData = {
        'isFit': is_Weight_In_between,
        'message': message
    }

    return jsonify( { 'fitInfo': fitData } ), 201

app.run(debug=True)
