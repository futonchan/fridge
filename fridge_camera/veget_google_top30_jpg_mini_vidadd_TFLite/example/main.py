#  -------------------------------------------------------------
#   Copyright (c) Microsoft Corporation.  All rights reserved.
#  -------------------------------------------------------------
"""
Skeleton code showing how to load and run the TensorFlow Lite export package from Lobe.
"""

import argparse
import json
import os
import RPi.GPIO as GPIO
import time
import sqlite3
import pigpio
import shutil
import sys
import signal

import cv2
from PIL import Image

from utils.tflite_model import TFLiteModel

from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
import threading
import time

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///maindb.db"
db = SQLAlchemy(app)

class Vegets(db.Model):
    # テーブル名
    __tablename__ = 'Vegets'

    # カラム情報
    # id = db.Column(db.Integer, primary_key=True)
    veget = db.Column(db.String(100), nullable=False, primary_key=True)
    count = db.Column(db.Integer, nullable=False)

    def to_dict(self):
        return {
            # 'id': self.id,
            'veget': self.veget,
            'count': self.count
        }

    def __repr__(self): # 確認用にprintしてくれる
        return f"Veget(veget={veget}, count={count})"

@app.route("/", methods=["GET"])
def list_veget():
    vegets = Vegets.query.all()
    return jsonify({'vegets': [veget.to_dict() for veget in vegets]})


def sig_handler(signum, frame) -> None:
    sys.exit(1)

def pulse_in(pin, value=GPIO.HIGH, timeout=1.0):
    """
    ピンに入力されるパルスを検出します。
    valueをHIGHに指定した場合、pulse_in関数は入力がHIGHに変わると同時に時間の計測を始め、
    またLOWに戻るまでの時間(つまりパルスの長さ)をマイクロ秒単位(*1)で返します。
    タイムアウトを指定した場合は、その時間を超えた時点で0を返します。
    *1 pythonの場合はtimeパッケージの仕様により実装依存ですが、概ねnanosecで返ると思います。
    :param pin: ピン番号、またはGPIO 番号(GPIO.setmodeに依存。)
    :param value: パルスの種類(GPIO.HIGH か GPIO.LOW。default:GPIO.HIGH)
    :param timeout: タイムアウト(default:1sec)
    :return: パルスの長さ（秒）タイムアウト時は0
    """
    start_time = time.time()
    not_value = (not value)

    # 前のパルスが終了するのを待つ
    while GPIO.input(pin) == value:
        if time.time() - start_time > timeout:
            return 0

    # パルスが始まるのを待つ
    while GPIO.input(pin) == not_value:
        if time.time() - start_time > timeout:
            return 0

    # パルス開始時刻を記録
    start = time.time()

    # パルスが終了するのを待つ
    while GPIO.input(pin) == value:
        if time.time() - start_time > timeout:
            return 0

    # パルス終了時刻を記録
    end = time.time()

    return end - start


def init_sensors(trig, echo, mode=GPIO.BCM):
    """
    初期化します
    :param trig: Trigger用ピン番号、またはGPIO 番号
    :param echo: Echo用ピン番号、またはGPIO 番号
    :param mode: GPIO.BCM、または GPIO.BOARD (default:GPIO.BCM)
    :return: なし
    """
    GPIO.cleanup()
    GPIO.setmode(mode)
    GPIO.setup(trig, GPIO.OUT)
    GPIO.setup(echo, GPIO.IN)


def get_distance(trig, echo, temp=15):
    """
    距離を取得します。取得に失敗した場合は0を返します。
    :param trig: Trigger用ピン番号、またはGPIO 番号(GPIO.setmodeに依存。)(GPIO.OUT)
    :param echo: Echo用ピン番号、またはGPIO 番号(GPIO.setmodeに依存。)(GPIO.IN)
    :param temp: 取得可能であれば温度(default:15)
    :return: 距離（ｃｍ）タイムアウト時は 0
    """

    # 出力を初期化
    GPIO.output(trig, GPIO.LOW)
    time.sleep(0.3)
    # 出力(10us以上待つ)
    GPIO.output(trig, GPIO.HIGH)
    time.sleep(0.000011)
    # 出力停止
    GPIO.output(trig, GPIO.LOW)

    # echo からパルスを取得
    dur = pulse_in(echo, GPIO.HIGH, 1.0)

    # ( パルス時間 x 331.50 + 0.61 * 温度 ) x (単位をcmに変換) x 往復
    # return dur * (331.50 + 0.61 * temp) * 100 / 2
    ret = dur * (331.50 + 0.61 * temp) * 50

    return ret

def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

def main():
    model_dir = os.path.join(os.getcwd(), "..")


    GPIO_TRIG = 26
    GPIO_ECHO = 19

    init_sensors(GPIO_TRIG, GPIO_ECHO)

    INPUT_DBNAME = "maindb.db"
    OUTPUT_DBNAME = "/var/www/html/maindb_copy.db"
    conn = sqlite3.connect(INPUT_DBNAME)
    cur = conn.cursor() # cursorインスタンスからDB操作する


    cur.execute("SELECT * FROM Vegets")
    print(cur.fetchone()) #...[8]

    gpio_pin0 = 18

    pi = pigpio.pi()
    pi.set_mode(gpio_pin0, pigpio.OUTPUT)

    fridge_max = 40

    cap = cv2.VideoCapture(0)
    model = TFLiteModel(model_dir)
    model.load()

    try:
        while True:
            while True:
                ret, frame = cap.read()
                print(ret)
                distance_cm = get_distance(GPIO_TRIG, GPIO_ECHO)
                if distance_cm > fridge_max:
                    print("Out of fridge_max")
                elif distance_cm > 9 and distance_cm < 19:
                    cv2.imwrite("test.jpg", frame)
                    print("距離：{0} cm".format(distance_cm))
                    break
                else:
                    print("距離：{0} cm".format(distance_cm))
                time.sleep(1)

            image = cv2pil(frame)
            outputs = model.predict(image)
            print(f"Predicted: {outputs}")

            top_veget_name = outputs["predictions"][0]["label"]
            print("top_veget_name : {}".format(top_veget_name))
            cur.execute('SELECT count FROM Vegets WHERE veget = ? ', (top_veget_name,)) #...[7]
            row = cur.fetchone() #...[8]
            if row is None: #...[9]
                cur.execute('''INSERT INTO Vegets (veget, count)
                        VALUES (?, 1)''', (top_veget_name,))
            else: #...[10]
                cur.execute('UPDATE Vegets SET count = count + 1 WHERE veget = ?',
                            (top_veget_name,))
            conn.commit() #...[11]

            cur.execute("SELECT * FROM Vegets")
            print(cur.fetchall()) #...[8]

            shutil.copy(INPUT_DBNAME, OUTPUT_DBNAME)
            print("Copyed {0} to {1}.".format(INPUT_DBNAME, OUTPUT_DBNAME))

            sys.stdout.flush() # 明示的にflush

            pi.hardware_PWM(gpio_pin0, 523, 600000)
            time.sleep(1)
            pi.hardware_PWM(gpio_pin0, 0, 0)


    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        pi.set_mode(gpio_pin0, pigpio.INPUT)
        pi.stop()
        GPIO.cleanup()
        cap.release()
        cur.close()
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)


if __name__ == "__main__":
    api_thread = threading.Thread(name='rest_service', target=app.run, args=('0.0.0.0',), kwargs=dict(debug=False))
    api_thread.start()
#     app.run("0.0.0.0", debug=False)
    sys.exit(main())
    api_thread.join()