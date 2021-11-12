import websockets
import asyncio

# The main function that will handle connection and communication 
# with the server
async def listen():
    #url = "ws://127.0.0.1:7890"
    url = "ws://190.162.196.242:25565"
    # Connect to the server
    async with websockets.connect(url) as ws:
        # Send a greeting message
        incomplete_word = "garban"
        await ws.send(incomplete_word)
        # Stay alive forever, listening to incoming msgs
        while True:
            msg = await ws.recv()
            print(msg)

# Start the connection
asyncio.get_event_loop().run_until_complete(listen())