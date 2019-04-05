import requests
from bs4 import BeautifulSoup

hopitals = []


url = "https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States"
source_code = requests.get(url)
plain_text = source_code.text

# Convert the plain text of the site into recognizable html
soup = BeautifulSoup(plain_text, "html.parser")