from flask import Flask
from flask_socketio import SocketIO
import flask
import torch
from sentence_transformers import SentenceTransformer
import numpy
import re
import nltk

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
data1 = ""
model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
vectors = torch.load('vector_db.pt')
length_of_reply = 5


def preprocessing():
    with open("db1.txt", "r", encoding="utf_8") as file:
        clean_file = file.read()
    clean_file = re.sub(r"\s+", " ", clean_file)
    clean_file = re.sub(r"\n+", "", clean_file)
    clean_file = re.sub(r"\\u00", "", clean_file)
    clean_file = clean_file.lower()
    corpus = nltk.sent_tokenize(clean_file)
    return corpus


async def answer_queries(input_):
    corpus = preprocessing()
    response = ""
    best_finds = {}
    # Encodes query
    print("Encoding Query")
    query = model.encode([input_])
    # Calculates cosine sim
    print("Calculating similarities. . .")
    similarities = model.similarity(query, numpy.array(vectors))
    print("Responding to query. . .")
    for index, sentence in enumerate(corpus):
        best_finds.update({sentence: int(similarities[0][index])})
    highest = max(list(best_finds.values()))
    if highest < 0.5:
        return "My apologies, but there is no data found on this subject."
    for n in range(length_of_reply):
        good_answer = max(best_finds, key=best_finds.get)
        if best_finds[good_answer] > highest - 0.25:
            response += good_answer + " "
            best_finds.pop(good_answer)
        else:
            break
    return response


@app.route("/", methods=['GET'])
async def index():
    global data1
    data1 = await answer_queries(data1)
    response = flask.jsonify({'result': data1})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@socketio.on("message")
def update(data):
    global data1
    data1 = data['value']['inputText']


if __name__ == "__main__":
    socketio.run(app, port=80, host='0.0.0.0', debug=True)
