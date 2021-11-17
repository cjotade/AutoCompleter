import os
import json
from fast_autocomplete import AutoComplete

from utils import get_words, load_data, create_model, get_best_response
from ModelManager import ModelManager

import websockets
import asyncio

async def handle(websocket, path):
    print("A client just connected")
    try:
        async for incomplete_word in websocket:
            print("Incomplete word from client: " + incomplete_word)
            best_response = get_best_response(incomplete_word=incomplete_word, model=model, **params)
            if best_response:
                await websocket.send(best_response)
    except websockets.exceptions.ConnectionClosed as e:
        print("A client just disconnected")

if __name__ == "__main__":
    # Load model
    model = create_model(data_folder="../data/")
    
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
