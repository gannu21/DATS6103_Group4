import requests
from bs4 import BeautifulSoup
url = 'https://www.carmax.com/transportation/locations'
headers = {
    'cookie': "uid=05191da8-670a-4a66-9352-cdfeee32dca6",
    'authority': "gum.criteo.com",
    'accept': "*/*",
    'accept-language': "en-US,en;q=0.9",
    'cache-control': "no-cache",
    'pragma': "no-cache",
    'referer': "https://gum.criteo.com/syncframe?topUrl=www.carmax.com&origin=onetag",
    'sec-ch-ua-mobile': "?0",
    'sec-fetch-dest': "empty",
    'sec-fetch-mode': "cors",
    'sec-fetch-site': "same-origin",
    'user-agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36"
    }

request = requests.get(url, headers=headers)
soup = BeautifulSoup(request.content, 'html5')


add = []
for i in soup.find_all('div', class_='transportation-locations-store-section'):
    add.append(i.find('p').text.strip())

for i in add:
    print(re.findall('\d{5}$', i))