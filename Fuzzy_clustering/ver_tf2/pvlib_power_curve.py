import pvlib
import numpy as np
import pandas as pd


def sol_pos(times_index,latitude,longitude,tz):
    times_index=pd.DatetimeIndex([times_index])
    altitude = 42  # above the sea level in meters

    sand_point = pvlib.location.Location(latitude, longitude, tz=tz, altitude=altitude, name=tz)

    solpos = pvlib.solarposition.get_solarposition(times_index, sand_point.latitude, sand_point.longitude)

    return solpos['apparent_zenith'].values[0]