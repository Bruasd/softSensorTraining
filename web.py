import asyncio
import websockets
import json

async def control(websocket, path):
    while True:
        # Example: Read the position and orientation from somewhere (e.g., a file, sensors, etc.)
        position_orientation = {
            "position":
                [1.0, 2.0, 3.0],
            "orientation": [0.0, 0.0, 0.0]
        }
        await websocket.send(json.dumps(position_orientation))
        await asyncio.sleep(1)  # Update every second

start_server = websockets.serve(control, "localhost", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()