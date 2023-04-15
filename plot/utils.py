from datetime import datetime
from suntime import Sun, SunTimeException

def get_sunrise_sunset(date, coords):
    sun = Sun(coords['lat'], coords['long'])
    abd = datetime.strptime(date, '%Y-%m-%d').date()
    
    try:
        sunrise = sun.get_sunrise_time(abd).time().strftime('%H')
        sunset = sun.get_sunset_time(abd).time().strftime('%H')

        return sunrise, sunset
    except SunTimeException as e:
        print(f"Error: {e}")

