from gpiozero import Button, LED

button = Button(18)
led = LED(17)

while True:
    if button.is_pressed:
        print("Button is pressed")
        led.on()
    else:
        print("Button is not pressed")
        led.off()
