from flask import Flask, render_template, request
import numpy as np
import pickle




# Load the model
with open('proj22.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Get the numerical inputs from the form
    store_sales =float( request.form['store_sales'])
    unit_sales = float(request.form['unit_sales'])
    promotion_name = float(request.form['promotion_name'])
    total_children = float(request.form['total_children'])
    num_children_at_home =float( request.form['num_children_at_home'])
    gross_weight = float(request.form['gross_weight'])
    low_fat =float( request.form['low_fat'])
    recyclable_package = float(request.form['recyclable_package'])
    units_per_case = float(request.form['units_per_case'])
    store_type = float(request.form['store_type'])
    store_city = float(request.form['store_city'])
    store_state = float(request.form['store_state'])
    store_sqft = float(request.form['store_sqft'])
    coffee_bar = float(request.form['coffee_bar'])
    video_store = float(request.form['video_store'])
    salad_bar = float(request.form['salad_bar'])
    prepared_food = float(request.form['prepared_food'])
    florist = float(request.form['florist'])
    media_type = float(request.form['media_type'])
    avg_cars_at_home = float(request.form['avg_cars_at_home'])
    # Create the final features array
    final_features = np.array([[avg_cars_at_home,media_type,florist,prepared_food,salad_bar,video_store,coffee_bar,store_sqft,store_state,store_city,store_type,units_per_case,recyclable_package,low_fat,gross_weight,num_children_at_home,total_children,promotion_name,unit_sales,store_sales]])

    # Make the prediction
    prediction = model.predict(final_features)[0]
    # Render the result template with the prediction text
    return render_template('submit.html', prediction_text=prediction)

if __name__ == '__main__':
    app.run(debug=True)