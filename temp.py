import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import folium
import imageio
from tqdm import tqdm_notebook
from folium.plugins import MarkerCluster
import geoplot as gplt
import geopandas as gpd
import geoplot.crs as gcrs
import imageio
import mapclassify as mc
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
import scipy
from itertools import product
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf

plt.style.use('ggplot')
plt.rcParams['font.family'] = 'sans-serif' 
plt.rcParams['font.serif'] = 'Ubuntu' 
plt.rcParams['font.monospace'] = 'Ubuntu Mono' 
plt.rcParams['font.size'] = 14 
plt.rcParams['axes.labelsize'] = 12 
plt.rcParams['axes.labelweight'] = 'bold' 
plt.rcParams['axes.titlesize'] = 12 
plt.rcParams['xtick.labelsize'] = 12 
plt.rcParams['ytick.labelsize'] = 12 
plt.rcParams['legend.fontsize'] = 12 
plt.rcParams['figure.titlesize'] = 12 
plt.rcParams['image.cmap'] = 'jet' 
plt.rcParams['image.interpolation'] = 'none' 
plt.rcParams['figure.figsize'] = (12, 10) 
plt.rcParams['axes.grid']=True
plt.rcParams['lines.linewidth'] = 2 
plt.rcParams['lines.markersize'] = 8
colors = ['xkcd:pale orange', 'xkcd:sea blue', 'xkcd:pale red', 'xkcd:sage green', 'xkcd:terra cotta', 'xkcd:dull purple', 'xkcd:teal', 'xkcd: goldenrod', 'xkcd:cadet blue',
'xkcd:scarlet']


data = pd.read_csv('GlobalLandTemperaturesByCity.csv')

city_data = data.drop_duplicates(['City'])

# from geopy.geocoders import Nominatim

# world_map= folium.Map()
# geolocator = Nominatim(user_agent="Piero")
# marker_cluster = MarkerCluster().add_to(world_map)

# def convert(coord):
# 		mult = 1 if coord[-1] in ['N','E'] else -1
# 		return mult * float(coord[:-1]) 

# for i in range(len(city_data)):
# 		lat = convert(city_data.iloc[i]['Latitude'])
# 		long = convert(city_data.iloc[i]['Longitude'])
# 		radius=5
# 		folium.CircleMarker(location = [lat, long], radius=radius,fill =True, color='darkred',fill_color='darkred').add_to(marker_cluster)   

# world_map.save('map.html')

# explodes = (0,0.3)
# plt.pie(data[data['City']=='Chicago'].AverageTemperature.isna().value_counts(),explode=explodes,startangle=0,colors=['firebrick','indianred'],
# 	labels=['Non NaN elements','NaN elements'], textprops={'fontsize': 20})

# plt.show()

chicago_d = data[data['City']=='Chicago']
# print(chicago_d.head(20))

chicago_d['AverageTemperature']=chicago_d.AverageTemperature.bfill()
print()
# print(chicago_d.head(20))

chicago_d['AverageTemperatureUncertainty']=chicago_d.AverageTemperatureUncertainty.bfill()
print()
# print(chicago_d.head(20))

