# Save this as app.py
from flask import Flask, render_template, request, Response, jsonify
import subprocess
import os
import sys

app = Flask(__name__)

# Path to your Python script
AUTHENTICATION_SCRIPT = "final.py"

@app.route('/')
def index():
    """Render the main page with the button"""
    return render_template('index.html')

@app.route('/run-authentication', methods=['POST'])
def run_authentication():
    """Run the authentication script when requested"""
    try:
        # Run the Python script as a subprocess
        process = subprocess.Popen(
            [sys.executable, AUTHENTICATION_SCRIPT],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Immediately return a success response
        return jsonify(status="success", message="Authentication process started")
    
    except Exception as e:
        # Return a proper JSON error response
        return jsonify(status="error", message=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True)
