"""NASA Horizons API integration for orbital mechanics simulations."""

import datetime
from typing import Optional
import numpy as np
import requests


def get_planet_vectors(start_time_str: str) -> np.ndarray:
    """Fetch planetary vectors from NASA's Horizons API.
    
    This function retrieves position and velocity vectors for the Sun and 8 planets
    from NASA's JPL Horizons system at a specified date.
    
    Parameters
    ----------
    start_time_str : str
        Start date in 'YYYY-MM-DD' format
        
    Returns
    -------
    np.ndarray
        Array of shape (9, 6) containing [x, y, z, vx, vy, vz] vectors
        for Sun, Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune
        
    Example
    -------
    >>> vectors = get_planet_vectors('2022-12-20')
    >>> vectors.shape
    (9, 6)
    """
    API_URL = 'https://ssd.jpl.nasa.gov/api/horizons.api'
    
    # Command codes for celestial bodies:
    # 10: Sun, 199: Mercury, 299: Venus, 399: Earth, 499: Mars,
    # 599: Jupiter, 699: Saturn, 799: Uranus, 899: Neptune
    command_codes = ['10', '199', '299', '399', '499', '599', '699', '799', '899']
    
    options = {
        "format": 'json',
        "MAKE_EPHEM": 'YES',
        "COMMAND": None,
        "EPHEM_TYPE": 'VECTORS',
        "CENTER": '500@0',  # Solar System Barycenter
        "START_TIME": None,
        "STOP_TIME": None,
        "STEP_SIZE": '2d',
        "VEC_TABLE": '2',
        "REF_SYSTEM": "ICRF",
        "REF_PLANE": "ECLIPTIC",
        "VEC_CORR": "NONE",
        "OUT_UNITS": 'au-d',  # AU for distance, days for time
        "VEC_LABELS": "YES",
        "VEC_DELTA_T": "NO",
        "CSV_FORMAT": "YES",
        "OBJ_DATA": "YES",
    }

    start_time = datetime.datetime.strptime(start_time_str, '%Y-%m-%d')
    stop_time = start_time + datetime.timedelta(days=1)
    
    options['START_TIME'] = start_time.strftime('%Y-%m-%d')
    options['STOP_TIME'] = stop_time.strftime('%Y-%m-%d')
    
    planet_vectors = []
    for code in command_codes:
        options['COMMAND'] = code
        response = requests.get(API_URL, params=options)
        data = response.json()['result']
        
        # Extract CSV data between $$SOE and $$EOE markers
        csv_data = data[data.find('$$SOE')+5:data.find('$$EOE')-1]
        # Strip any final commas and split the data into a list
        csv_data = csv_data.strip(',').split(',')
        # Remove the first 2 elements, which are time in two different formats
        csv_data = csv_data[2:]
        # Convert strings to floats
        csv_data = [float(x) for x in csv_data]
        planet_vectors.append(csv_data)
        
    return np.array(planet_vectors)


def get_solar_system_bodies() -> dict:
    """Get information about solar system bodies supported by the API.
    
    Returns
    -------
    dict
        Dictionary mapping body names to their Horizons command codes
    """
    return {
        'Sun': '10',
        'Mercury': '199', 
        'Venus': '299',
        'Earth': '399',
        'Mars': '499',
        'Jupiter': '599',
        'Saturn': '699', 
        'Uranus': '799',
        'Neptune': '899'
    }


def validate_date_format(date_str: str) -> bool:
    """Validate that a date string is in the correct format for the API.
    
    Parameters
    ----------
    date_str : str
        Date string to validate
        
    Returns
    -------
    bool
        True if the date format is valid, False otherwise
    """
    try:
        datetime.datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False