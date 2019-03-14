import requests
from bs4 import BeautifulSoup

# Scrape data from the Deep_Learning wiki page
url = "https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States"
source_code = requests.get(url)
plain_text = source_code.text

# Convert the plain text of the site into recognizable html
soup = BeautifulSoup(plain_text, "html.parser")
wiki_table = soup.findAll('table')  # grab the table tag since there should only be one
table_body = wiki_table[0].find('tbody')

print(table_body.text)

rows = table_body.findAll('tr')
row_string = ""
for row in rows:
    cols = row.findAll('td')
    for col in cols:
        row_string = row_string + "\t" + col.text.rstrip()
    row_string = row_string + "\n"

file2 = open('states.txt', 'w')
file2.write(row_string)
file2.close()

print(row_string)
