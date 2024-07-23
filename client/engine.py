import requests

URL = "http://server:8080/explain?prompt="

def get_red_code(intensity):
    """
    Get the ANSI escape code for the red color with a given intensity.
    Intensity should be between 0 and 255.
    """
    return f'\033[38;2;{str(intensity)};0;0m'

def explain(prompt, model):
    res = requests.get(URL+prompt+"&model="+model)
    if res.status_code == 200:
        return res.json()["message"]
