<a href="https://cognitiveclass.ai"><img src = "https://ibm.box.com/shared/static/9gegpsmnsoo25ikkbl4qzlvlyjbgxs5x.png" width = 400> </a>

<h1 align=center><font size = 5>IBM Applied Data Science Capstone Course by Coursera</font></h1>
<h2 align=center><font size = 4> Segmenting and Clustering Neighborhoods in Toronto </font></h2>

* Build a dataframe of the postal code of each neighborhood along with the borough's name in Toronto.
* Get the geographical coordinates (latitude & longitude) of the neighborhoods in Toronto
* Explore and cluster the neighborhoods in Toronto by replicating analysis done to New York data.

## Table of Contents

<div class="alert alert-block alert-info" style="margin-top: 20px">

1. [Download and Explore Dataset](#0)<br>
2. [Explore Neighborhoods in Toronto](#1)<br>
3. [Analyze Each Neighborhood](#2) <br>
4. [Cluster Neighborhoods](#3) <br>
5. [Examine Clusters](#4) <br>
6. [Summary Analysis](#5) <br>
</div>
<hr>

## Assignment Part 1.

## Import libraries


```python
# library to handle vectorized data
import numpy as np

# library for data analysis
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

#library to handle JSON files
import json

#!pip install geopy

# convert an address into latitude and longitude values
from geopy.geocoders import Nominatim

# library to handle requests
import urllib.request

# library to handle requests
import requests

# library to for pulling and parsing data out of HTML and XML files
import bs4 as bs

# transform JSON into a pandas dataframe
from pandas.io.json import json_normalize

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

# map rendering library
#!pip install folium
import folium

print ("Libraries imported")
```

    Libraries imported
    

## 1. Download and Explore Dataset <a id="0"></a>


Use the Notebook to build the code to scrape the following Wikipedia page, https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M, in order to obtain the data that is in the table of postal codes and to transform the data into a pandas dataframe


### Data Wrangling 


```python
# Get data from Wikipedia page and convert to table

source = urllib.request.urlopen('https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M').read()
soup = bs.BeautifulSoup(source, 'lxml')
table = soup.find('table')
table_rows = table.find_all('tr')

# create lists to hold table columns data

postalcodeList = []
boroughList = []
neighborhoodList =[]

for tr in table_rows:
    td = tr.find_all('td')
    row = [i.text for i in td]
    if (len(row) > 0):
        postalcodeList.append(row[0].rstrip('\n'))
        boroughList.append(row[1].rstrip('\n'))
        neighborhoodList.append(row[2].rstrip('\n'))

# Create dataframe with three columns: PostalCode, Borough, and Neighborhood

df = pd.DataFrame({"PostalCode":postalcodeList,
                      "Borough": boroughList,
                 "Neighborhood":neighborhoodList})

# Only process the cells that have an assigned borough.
# Ignore cells with a borough that is Not assigned.

df_dropna = df[df.Borough != "Not assigned"].reset_index(drop = True)
df_dropna.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M3A</td>
      <td>North York</td>
      <td>Parkwoods</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M4A</td>
      <td>North York</td>
      <td>Victoria Village</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Regent Park, Harbourfront</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M6A</td>
      <td>North York</td>
      <td>Lawrence Manor, Lawrence Heights</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M7A</td>
      <td>Downtown Toronto</td>
      <td>Queen's Park, Ontario Provincial Government</td>
    </tr>
  </tbody>
</table>
</div>



###  Group neighborhoods within same borough.

More than one neighborhood can exist in one postal code area.
For example, in the table on the Wikipedia page, you will notice that M5A is listed twice and has two neighborhoods: Harbourfront and Regent Park. These two rows will be combined into one row with the neighborhoods separated with a comma.


```python
# group neighborhoods in the same borough

df_grouped = df_dropna.groupby(["PostalCode","Borough"], as_index=False).agg(lambda x:",".join(x))
df_grouped.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Malvern, Rouge</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>Scarborough</td>
      <td>Rouge Hill, Port Union, Highland Creek</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>Scarborough</td>
      <td>Guildwood, Morningside, West Hill</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>Scarborough</td>
      <td>Woburn</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>Scarborough</td>
      <td>Cedarbrae</td>
    </tr>
  </tbody>
</table>
</div>



###  Make "Not assigned" Neighborhoods value equal to Borough

If a cell has a borough but a "Not assigned" neighborhood, then the neighborhood will be the same as the borough.


```python
# Make "Not assigned" Neighborhoods value equal to Borough

for index, row in df_grouped.iterrows():
    if row["Neighborhood"] == "Not assigned":
        row["Neighborhood"] = row["Borough"]
df_grouped.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Malvern, Rouge</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>Scarborough</td>
      <td>Rouge Hill, Port Union, Highland Creek</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>Scarborough</td>
      <td>Guildwood, Morningside, West Hill</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>Scarborough</td>
      <td>Woburn</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>Scarborough</td>
      <td>Cedarbrae</td>
    </tr>
  </tbody>
</table>
</div>



### Print number of rows in clean dataframe

In the last cell of your notebook, use the .shape method to print the number of rows of your dataframe.


```python
# print number of rows in the clean dataframe

df_grouped.shape
```




    (103, 3)


