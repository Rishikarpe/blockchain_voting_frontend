from flask import Flask, request, jsonify
import csv

app = Flask(__name__)

# Define the path to the CSV file
CSV_FILE = 'users.csv'

# Function to save user details to CSV
def save_to_csv(user_data):
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([user_data['name'], user_data['email'], user_data['password']])

@app.route('/register', methods=['POST'])
def register():
    # Get form data from the request
    user_data = request.get_json()

    # Save the data to CSV
    save_to_csv(user_data)

    return jsonify({"message": "Registration successful!"}), 201

if __name__ == '__main__':
    app.run(debug=True)