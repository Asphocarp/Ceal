#! /usr/bin/env python3
import requests
import json

# accept param of file path
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-b", "--buffer", type=str, default="buf")
args, unknown = parser.parse_known_args()
FILE_PATH = f'temp/{args.buffer}.py'
print(f"Sending file: {FILE_PATH}")

# configuration
SERVER_URL = "http://localhost:5000/run"
API_PASSWORD = "your_secure_password_here"


def send_code_to_server(file_path):
    try:
        # Read the code from the file
        with open(file_path, "r") as file:
            code = file.read()

        # Create the request payload
        payload = {"code": code}

        # Set the headers, including the Authorization token
        headers = {
            "Authorization": f"Bearer {API_PASSWORD}",
            "Content-Type": "application/json"
        }

        # Send the POST request
        response = requests.post(SERVER_URL, headers=headers, json=payload)

        # Handle the response
        if response.status_code == 200:
            print(json.dumps(response.json(), indent=4))
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    send_code_to_server(FILE_PATH)