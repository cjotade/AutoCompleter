import websockets
import asyncio

# The main function that will handle connection and communication 
# with the server
async def listen():
    url = "ws://0.0.0.0:3000"
    # Connect to the server
    async with websockets.connect(url) as websocket:
        # Send a greeting message
        incomplete_word = input("word:")
        await websocket.send(incomplete_word)
        # Stay alive forever, listening to incoming msgs
        while True:
            msg = await websocket.recv()
            print(msg)

# Start the connection
asyncio.get_event_loop().run_until_complete(listen())