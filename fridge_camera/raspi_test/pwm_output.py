import pigpio
import time

gpio_pin0 = 18

pi = pigpio.pi()
pi.set_mode(gpio_pin0, pigpio.OUTPUT)

# GPIO18: 2Hz、duty比0.5
pi.hardware_PWM(gpio_pin0, 523, 600000)

time.sleep(1)
pi.set_mode(gpio_pin0, pigpio.INPUT)

pi.stop()



# import RPi.GPIO as GPIO
# import time

# GPIO_BUZZ = 18
# GPIO.setmode(GPIO.BOARD)

# GPIO.setup(GPIO_BUZZ, GPIO.OUT)
# p = GPIO.PWM(GPIO_BUZZ, 300)
# p.start(50)

# time.sleep(1)

# p.stop()
# GPIO.cleanup()

# import RPi.GPIO as GPIO
# GPIO.setmode(GPIO.BOARD)
# GPIO.setup(18, GPIO.OUT)

# p = GPIO.PWM(18, 0.5)
# p.start(1)
# input('停止するにはEnterキーを押す:')   #
# p.stop()
# GPIO.cleanup()