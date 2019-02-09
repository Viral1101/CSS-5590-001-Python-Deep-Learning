import requests
from bs4 import BeautifulSoup

# Scrape data from the Deep_Learning wiki page
url = "https://en.wikipedia.org/wiki/Deep_learning"
source_code = requests.get(url)
plain_text = source_code.text

# Convert the plain text of the site into recognizable html
soup = BeautifulSoup(plain_text, "html.parser")
result_list = soup.find('title')  # grab the title tag since there should only be one
print(result_list.text)  # print out the text of the tag

links = soup.findAll('a')   # grab a list of all the anchor tags
for link in links:          # iterate over all the anchor tags
    if link.get('href'):        # proceed if the anchor tag contains a reference (will crash if a tag is missing href)
        if link.get('href').startswith("http"):     # get all the web links
            href = link.get('href')
            print(href)                             # display the links
