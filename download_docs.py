import requests
from bs4 import BeautifulSoup
import os
import urllib

# The URL to scrape
url = "https://gpt-index.readthedocs.io/en/stable/"
url = "https://docs.llamaindex.ai/en/stable/"
url = "https://llama-index.readthedocs.io/zh/stable/"

# The directory to store files in
output_dir = "./docs/"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Fetch the page
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find all links to .html files
links = soup.find_all('a', href=True)
# print (f"links: {links}")
href_count = 1
endswith_count = 1
for link in links:
    href = link['href']
    # print ("href_count: %s" % (href_count,))
    href_count += 1
    # If it's a .html file
    if href.endswith('.html'):
        # Make a full URL if necessary
        if not href.startswith('http'):
            href = urllib.parse.urljoin(url, href)
        print("endswith_count: %s" % (endswith_count,))
        endswith_count += 1
        # Fetch the .html file
        print(f"downloading {href}")
        file_response = requests.get(href)

        # Write it to a file
        file_name = os.path.join(output_dir, os.path.basename(href))
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(file_response.text)

# print ("href_count: %s" % (href_count,))
# print ("endswith_count: %s" % (endswith_count,))
# print (links)
# print (type(links))