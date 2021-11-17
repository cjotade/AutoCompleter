import os
import json
from fast_autocomplete import AutoComplete

from utils import get_words
from ModelManager import ModelManager

import websockets
import asyncio


def load_data(data_folder="../data/"):
    filename_train = "ES-es_colloquial_cleaned.txt"
    with open(os.path.join(data_folder, filename_train), 'r') as fp:
        data_train = fp.read().splitlines()

    filename_test = "ES-es_colloquial_01_cleaned.txt"
    with open(os.path.join(data_folder, filename_test), 'r') as fp:
        data_test = fp.read().splitlines()

    filename_spanish = 'spanish_words.json'
    with open(os.path.join(data_folder, filename_spanish), 'r') as fp:
        spanish_cnt_words = json.load(fp)
    
    return data_train, data_test, spanish_cnt_words

def create_model(data, spanish_cnt_words=None):
    # Words and count
    words, cnt_words, ngrams_words, _, _, _ = get_words(data)
    # Suffix Tree Model
    st_autocompleter = AutoComplete(words=ngrams_words)
    # Metrics calculator
    return ModelManager(st_autocompleter, cnt_words, spanish_count_words=spanish_cnt_words)  

def get_best_response(incomplete_word, model, params={}):
    # Calculate predictions
    predictions = model.predict(
        incomplete_word=incomplete_word,
        **params
    )
    if predictions:
        best_response, responses, dist_responses = predictions
        return best_response
    else:
        return None

async def handle(websocket, path):
    print("A client just connected")
    try:
        async for incomplete_word in websocket:
            print("Incomplete word from client: " + incomplete_word)
            best_response = get_best_response(incomplete_word=incomplete_word, model=model, params=params)
            if best_response:
                await websocket.send(best_response)
    except websockets.exceptions.ConnectionClosed as e:
        print("A client just disconnected")

if __name__ == "__main__":
    # Load data
    data_train, data_test, spanish_cnt_words = load_data(data_folder="../data/")
    # Load model
    model = create_model(data_train, spanish_cnt_words=spanish_cnt_words)

    # Predict params
    params = {
        "max_cost": 1000,
        "size_search": 1000,
        "max_dist_threshold": 0.4,
        "scale_mode": "empirical",
        "no_response_use_sm_priors": False,
    }

    PORT = 3000
    print("Server listening on Port " + str(PORT))
    start_server = websockets.serve(handle, "0.0.0.0", PORT)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
