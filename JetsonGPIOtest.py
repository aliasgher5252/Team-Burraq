import Jetson.GPIO as GPIO
import time

# Pin Definitions
output_pin = 13  # BOARD pin 12

def main():
    # Pin Setup:
    GPIO.setmode(GPIO.BOARD)  # BCM pin-numbering scheme from Raspberry Pi
    # set pin as an output pin with optional initial state of HIGH
    GPIO.setup(output_pin, GPIO.OUT, initial=GPIO.HIGH)
    a = input("Press Enter to continue")
    print("Starting demo now! Press CTRL+C to exit")
            # Toggle the output every second
    print("Waiting")
    GPIO.output(output_pin, GPIO.HIGH)
    time.sleep(10)
    print("Outputting")
    GPIO.output(output_pin, GPIO.LOW)
    GPIO.cleanup()

if __name__ == '__main__':
    main()
