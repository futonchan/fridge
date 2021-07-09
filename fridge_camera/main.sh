sudo pigpiod
cd ~/prog/veget_class/fridge_camera/veget_google_top30_jpg_mini_vidadd_TFLite/example
source tflite-venv/bin/activate
# nohup python tflite_example.py  > tflite_example.log &
python main.py
