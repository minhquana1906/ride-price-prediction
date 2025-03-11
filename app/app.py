from flask import Flask, jsonify

app = Flask(__name__)


@app.route("/", methods=["GET"])
def read_root():
    return jsonify({"message": "Hello, World!"})


# @app.route('/predict', methods=['POST'])


if __name__ == "__main__":
    app.run(host="localhost", port=5001)
