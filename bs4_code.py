import pandas as pd 
data = pd.read_csv('D:\ADITYA\Documents\DataHack Datathon\Dataset\car_dataset_temp.csv') 
brands = [] 
models = [] 

for i, j in zip(data['Brand'], data['Model']): 
    brands.append(i) 
    models.append(j)

from bs4 import BeautifulSoup
import requests
c = 0
price_range = []
for i,j in zip(brands,models):
  try:
    baseurl = "https://www.cartrade.com/{i}-cars/{j}/".format(i=i,j=j)
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'}
    response = requests.get(baseurl, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    temp = soup.find('p', class_= "priceblock-price js-version-price").getText()
    if len(temp)==0: 
       price_range.append(data['Price'][0]) 
       continue
    price_range.append(temp)
  except: 
    price_range.append(data['Price'][c])
    continue
  c += 1  

print(price_range)