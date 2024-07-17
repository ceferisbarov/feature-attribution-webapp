import requests

URL = "http://server:8080/explain?prompt="

def get_red_code(intensity):
    """
    Get the ANSI escape code for the red color with a given intensity.
    Intensity should be between 0 and 255.
    """
    return f'\033[38;2;{str(intensity)};0;0m'

def explain(prompt):
    res = requests.get(URL+prompt)
    print("========================================")
    print("========================================")
    print(res.status_code)
    print(res.text)
    print("========================================")
    print("========================================")
    return res.json()["message"]
