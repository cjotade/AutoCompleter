import os
import json
from utils import  create_model
import websockets
import asyncio

# The main function that will handle connection and communication 
# with the server
async def listen():
    url = "ws://0.0.0.0:3000"
    # Connect to the server
    async with websockets.connect(url) as websocket:
        print()
        print("##################### Testing #####################")
        print(f"Checking errors in {len(decisions)} decisions")
        errors = []
        for i, decisions_dict in enumerate(decisions):
            # Get decisions from model
            word = decisions_dict["word"]
            best_response = decisions_dict["best_response"]
            correct_decision = decisions_dict["correct_decision"]
            incomplete_word = decisions_dict["incomplete_word"]
            # Send incomplete_word to server 
            await websocket.send(incomplete_word)
            # Get completed word from server
            word_completed = await websocket.recv()
            print(f"Checking {word_completed}, {word}, {best_response}")
            if correct_decision:
                if word != word_completed:
                    errors.append({
                        "word": word,
                        "incompleted_word": incomplete_word,
                        "word_completed": word_completed
                    })
            else:
                if best_response != word_completed:
                    errors.append({
                        "word": word,
                        "incompleted_word": incomplete_word,
                        "word_completed": word_completed
                    })
            if i % 50 == 0:
                print(f"Errors detected: {len(errors)}")
                print(f"Already checked words: {i}")
        print("Saving results in '../results/test_server_results.json'")
        if not os.path.exists('../results/'):
            os.makedirs('../results/')
        with open('../results/test_server_results.json', 'w') as fp:
            json.dump(errors, fp)

if __name__ == "__main__":
    data_folder = "../data/"
    # Load model
    print("Loading model...")
    m = create_model(data_folder=data_folder)
    # Predict params   
    params = {
        "max_cost": 1000,
        "size_search": 1000,
        "max_dist_threshold": 0.4,
        "scale_mode": "empirical",
        "no_response_use_sm_priors": False,
    }
    print("Loading data...")
    filename_test = "ES-es_colloquial_01_cleaned.txt"
    with open(os.path.join(data_folder, filename_test), 'r') as file:
        data_test = file.read().splitlines()
    print("Calculating correct decisions...")
    # Calculate found_words and non_found_words
    found_words, non_found_words, words_predicted_by_line = m.fill_metrics(
        data_test,
        **params
    )
    decisions = []
    for fw in found_words:
        if fw["correct_decision"]:
            #correct_decisions.append(fw)
            decisions.append({
                "word": fw["word"],
                "incomplete_word": fw["incomplete_word"],
                "best_response": fw["best_response"],
                "correct_decision": fw["correct_decision"]
            })
    # Start the connection
    print("Starting connection...")
    asyncio.get_event_loop().run_until_complete(listen())