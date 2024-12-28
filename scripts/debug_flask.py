from flask import Flask, request, jsonify
import subprocess
import os
import time
import io
import contextlib
import sys


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


app = Flask(__name__)

# Directory for temporary code execution
EXECUTION_DIR = "temp"
os.makedirs(EXECUTION_DIR, exist_ok=True)

# Password for accessing the API
API_PASSWORD = "your_secure_password_here"

# the persistent globals and locals
persistent_globals = globals().copy()
persistent_locals = locals().copy()

@app.before_request
def check_password():
    """Middleware to check for the correct password in headers."""
    auth_header = request.headers.get("Authorization")
    if auth_header != f"Bearer {API_PASSWORD}":
        return jsonify({"error": "Unauthorized"}), 401

@app.route('/clear', methods=['POST'])
def clear_persistent():
    persistent_globals.clear()
    persistent_locals.clear()
    return jsonify({"message": "Globals and locals cleared"})

@app.route('/run', methods=['POST'])
def run_code():
    # Get the code from the request
    data = request.get_json()
    if 'code' not in data:
        return jsonify({"error": "No code provided"}), 400

    code = data['code']

    # Write the code to a file for logging
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    file_name = f"{timestamp}.py"
    file_path = os.path.join(EXECUTION_DIR, file_name)
    with open(file_path, 'w') as f:
        f.write(code)

    # Prepare the log file
    log_file = f"temp/{timestamp}.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    log_fh = open(log_file, 'a')

    # Redirect stdout and stderr to both the terminal and the log file
    tee_stdout = Tee(sys.stdout, log_fh)
    tee_stderr = Tee(sys.stderr, log_fh)

    try:
        # Redirect stdout and stderr
        with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
            exec(code, persistent_globals, persistent_locals)

        return jsonify("done")

    except Exception as e:
        log_fh.write(f"Error: {str(e)}\n")
        return jsonify({"error": str(e)}), 500

    finally:
        log_fh.close()

@app.route('/run_shell', methods=['POST'])
def run_shell():
    data = request.get_json()
    if 'code' not in data:
        return jsonify({"error": "No code provided"}), 400
    code = data['code']
    result = subprocess.run(code, shell=True, capture_output=True, text=True)
    return jsonify({"output": result.stdout, "error": result.stderr})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)