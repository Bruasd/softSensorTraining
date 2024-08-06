from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import random
import json
import threading
import time

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")  # 允许所有来源

@app.route('/')
def index():
    return render_template('index.html')

def generate_random_data():
    while True:
        force = random.uniform(3, 3.5)
        angle = random.uniform(0, 360)  # 模拟随机角度
        x = random.uniform(-5, 5)  # 模拟随机 X 坐标
        y = random.uniform(-5, 5)  # 模拟随机 Y 坐标
        data = {
            'force': round(force, 2),
            'angle': round(angle, 2),
            'x': round(x, 2),
            'y': round(y, 2)
        }
        print(data)
        socketio.emit('force_data', data)
        time.sleep(0.2)

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    threading.Thread(target=generate_random_data).start()
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)