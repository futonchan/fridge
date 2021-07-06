import cv2
import datetime


def check_camera_connection():
    """
    Check the connection between any camera and the PC.

    """

    print('[', datetime.datetime.now(), ']', 'searching any camera...')
    true_camera_is = []  # 空の配列を用意

    # カメラ番号を0～9まで変えて、COM_PORTに認識されているカメラを探す
    for camera_number in range(0, 10):
        cap = cv2.VideoCapture(camera_number)
        ret, frame = cap.read()

        if ret is True:
            true_camera_is.append(camera_number)
            print("camera_number", camera_number, "Find!")

        else:
            print("camera_number", camera_number, "None")
    print("接続されているカメラは", len(true_camera_is), "台です。")
    return true_camera_is

def get_camera_propaties(camera):
    print("Checking camera id {} params.".format(camera))
    params = ['MSEC',
            'POS_FRAMES',
            'POS_AVI_RATIO',
            'FRAME_WIDTH',
            'FRAME_HEIGHT',
            'PROP_FPS',
            'PROP_FOURCC',
            'FRAME_COUNT',
            'FORMAT',
            'MODE',
            'BRIGHTNESS',
            'CONTRAST',
            'SATURATION',
            'HUE',
            'GAIN',
            'EXPOSURE',
            'CONVERT_RGB',
            'WHITE_BALANCE',
            'RECTIFICATION']

    cap = cv2.VideoCapture(camera)
    for num in range(19):
        print(params[num], ':', cap.get(num))


if __name__ == "__main__":
    alive_cameras = check_camera_connection()
    for cam in alive_cameras:
        get_camera_propaties(cam)
