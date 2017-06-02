#This scripts makes a snapshot of  the information available
# for the bird to define its context
#Sound is not considered but time, location, weather and light levels are

import time
import datetime
import pyowm
import requests
import json
import RPi.GPIO as GPIO
from geopy.geocoders import Nominatim
geolocator = Nominatim()

GPIO.setmode(GPIO.BOARD)
#define the pin that goes to the circuit
pin_to_circuit = 7


def get_time():
    #calculate time
    ts = time.time()
    h=datetime.datetime.fromtimestamp(ts).strftime('%H')
    m=datetime.datetime.fromtimestamp(ts).strftime('%M')
    s=datetime.datetime.fromtimestamp(ts).strftime('%S')
    time_now=(h,m,s)
    return time_now

def get_coordinates():
    #get location
    send_url = 'http://freegeoip.net/json'
    r = requests.get(send_url)
    j = json.loads(r.text)
    lat = j['latitude']
    lon = j['longitude']
    coord=(lat,lon);
    location = geolocator.reverse(coord,timeout=10)

    return coord

def get_weather():
    #get weather
    coord=get_coordinates()
    #need to find a weather api with no key needed
    owm = pyowm.OWM('4eb484c62d87888339ec463be1e66922')
    obs = owm.weather_at_coords(coord[0],coord[1])
    w = obs.get_weather()
    status=w.get_detailed_status()
    return status

def get_light(pin_to_circuit):
    count = 0

    #Output on the pin for
    GPIO.setup(pin_to_circuit, GPIO.OUT)
    GPIO.output(pin_to_circuit, GPIO.LOW)
    time.sleep(0.1)

    #Change the pin back to input
    GPIO.setup(pin_to_circuit, GPIO.IN)

    #Count until the pin goes high
    while (GPIO.input(pin_to_circuit) == GPIO.LOW):
        count += 1

    return count

def get_avg_light(sample_size):
    sample=0
    light_val=[]
    while (sample<sample_size):
        val=get_light(pin_to_circuit)
        light_val.append(val)
        sample=sample+1
        #print val
    else:
        avg=sum(light_val)/len(light_val)

    return avg

try:
    print get_time()
    print get_coordinates()
    print "It was %s today" % get_weather()
    #print get_light(pin_to_circuit)
    print "The light was level was %d" % get_avg_light(50)
except KeyboardInterrupt:
    print "Too fast man"
finally:
    GPIO.cleanup()
