import numpy as np
import pandas as pd
import tensorflow
import tflearn
import geojson
import sys
import joblib
from sklearn.metrics import *
from datetime import datetime
import rasterio
from collections import OrderedDict
from netCDF4 import Dataset
from geotiff import GeoTiff

#loading the solar farms data(for training)
with open("solar_farms_india_2021.geojson") as f:
	gj = geojson.load(f)
ft = gj['features'][0]

#Loading the soil data
# def read_netcdf(netcdf_file):
# 	contents = OrderedDict()
# 	data = Dataset(netcdf_file, 'r')
# 	for var in data.variables:
# 		attrs = data.variables[var].ncattrs()
# 		if attrs:
# 			for attr in attrs:
# 				print(''),\
# 					  repr(data.variables[var].getncattr(attr))
# 		contents[var] = data.variables[var][:]
# 	data = contents['precip']
# 	if len(data.shape) == 3:
# 		data = data.swapaxes(0,2)
# 		data = data.swapaxes(0,1)
# 		return data
# 	else:
# 		return data
	
# sdata = read_netcdf("..\data\CCS_India_2024-04-15083630am.nc")

#Loading the solar intensity data
print("Loading Solar Intensity Data...")
ghi = rasterio.open('GHI.tif')
ghidata = ghi.read(1)
smin = ghidata.min()
smax = ghidata.max()

#Loading the temperature data
print("Loading Temperature Data...")
tempr = rasterio.open('TEMP.tif')
tempdata = tempr.read(1)
tmin = tempdata.min()
tmax = tempdata.max()

#Loading the precipitation data
print("Loading Precipitation Data...")
prec = rasterio.open('PERSIANN_1y2020.tif')
precdata = prec.read(1)
pmin = precdata.min()
pmax = precdata.max()

#Creating the temperature, solar intensity and preciptation datasets
temp = []
precip = []
solari = []
# soil = []

lobox1 = np.linspace(81.66, 85.11, 25)
lobox2 = np.linspace(75.11, 78.58, 25)
lobox3 = np.linspace(91.38, 94.83, 25)
lobox4 = np.linspace(73.04, 73.38, 25)

labox1 = np.linspace(20.16, 21.53, 25)
labox2 = np.linspace(30.56, 32.01, 25)
labox3 = np.linspace(25.21, 26.75, 25)
labox4 = np.linspace(16.6, 19.15, 25)

logrid1, lagrid1 = np.meshgrid(lobox1, labox1)
logrid2, lagrid2 = np.meshgrid(lobox2, labox2)
logrid3, lagrid3 = np.meshgrid(lobox3, labox3)

labox = np.concatenate([lagrid1.flatten(), lagrid2.flatten(), lagrid3.flatten(), ])
lobox = np.concatenate([logrid1.flatten(), logrid2.flatten(), logrid3.flatten(), ])

labels = np.zeros((4158 + len(labox), 2))

print('Creating Training Dataset...')
for i in range(4158):
	long = gj['features'][i]['properties']['Longitude']
	lat = gj['features'][i]['properties']['Latitude']

	trow, tcol = tempr.index(long, lat)
	prow, pcol = prec.index(long, lat)
	srow, scol = ghi.index(long, lat)

	tval = tempdata[trow, tcol]
	pval = precdata[prow, pcol]
	sval = ghidata[srow, scol]

	temp.append(tval)
	precip.append(pval)
	solari.append(sval)
	labels[i][0] = 1

for i in range(len(labox)):
	long = lobox[i]
	lat = labox[i]	

	trow, tcol = tempr.index(long, lat)
	prow, pcol = prec.index(long, lat)
	srow, scol = ghi.index(long, lat)

	tval = tempdata[trow, tcol]
	pval = precdata[prow, pcol]
	sval = ghidata[srow, scol]

	temp.append(tval)
	precip.append(pval)
	solari.append(sval)
	labels[i+4158][1] = 1

#Creating the training data

# xtraining.append(precip[:4038] + precip[4158:4158 + len(labox)-60]) 
# xtraining.append(solari[:4038] + solari[4158:4158 + len(labox)-60])


# xtest = []
# xtest.append(temp[4038:4158] + temp[4158 + len(labox)-60:])
# xtest.append(precip[4038:4158] + precip[4158 + len(labox)-60:])
# xtest.append(solari[4038:4158] + solari[4158 + len(labox)-60:])

# ytrain = np.concatenate([labels[:4038], labels[4158:4158 + len(labox)-60]])
# ytest = np.concatenate([labels[4038:4158], labels[4158 + len(labox)-60:]])

training = []
training.append(temp)
training.append(precip)
training.append(solari)

ntrain = np.array(training)
print(ntrain.shape)
ntrain = ntrain.transpose()
print(ntrain.shape)
# ntest = np.array(xtest)
# ntest = ntest.transpose()
# ntest = np.array(xtest)
# ntest = ntest.transpose()
# sys.exit()
# print(training)
# print(labels)

# sys.exit()
# nlabels = np.array(labels)
# print(nlabels.shape)
# nlabels = nlabels.reshape(len(nlabels), 1)
# print(nlabels.shape)
# sys.exit()
print("Training the model")
net = tflearn.input_data(shape=[None, 3])
net = tflearn.fully_connected(net, 12)
net = tflearn.fully_connected(net, 12)
net = tflearn.fully_connected(net, 12)
net = tflearn.fully_connected(net, 12)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(ntrain, labels, n_epoch=30, show_metric=True)

# model.save("savedmodel.h5")
# print("Model Saved!")

# model = tensorflow.keras.models.load_model("savedmodel.h5")
# sys.exit()
# with open("pred_model.pickle", 'wb') as handle:
# 	pickle.dump(model, handle, pickle.HIGHEST_PROTOCOL)

# joblib.dump(model, 'pred_model.pkl')
model.save('model')

def inter():
	while True:
		la = input("Enter Latitude: ")
		lo = input("Enter Longitude: ")

		if lo.lower() == "quit":
			break
		if la.lower() == "quit":
			break
		
		laf = float(la)
		lof = float(lo)

		trow, tcol = tempr.index(lof, laf)
		prow, pcol = prec.index(lof, laf)
		srow, scol = ghi.index(lof, laf)
		
		ttd = tempdata[trow, tcol]
		ptd = precdata[prow, pcol]
		solard = ghidata[srow, scol]

		inp = np.array([ttd, ptd, solard])
		inp = inp.reshape(1, len(inp))

		res = model.predict(inp)
		
		print('Feasibilty: ', res[0][0]*100, '%')
	
# def test():
	# lonb = np.linspace(75.25, 78.5, 10)
	# latb = np.linspace(32.86, 34.83, 10)
	# res = np.zeros((len(ntest), 2))
	# for i in range(len(ntest)):
	# 	tval = ntest[i][0]
	# 	pval = ntest[i][1]
	# 	sval = ntest[i][2]

	# 	inp = np.array([tval, pval, sval])
	# 	inp = inp.reshape(1, len(inp))

	# 	res[i] = model.predict(inp)
	# 	print(res[i])
	# 	# print('Lat: ', lat, ' Long: ',long,' Label: ',labels[i + 4148],' Feasibility: ', res[0][0])
	
	# print(res.shape)
	# print(ytest.shape)
	# print('Accuracy Score: ', accuracy_score(res,ytest))
	# print('F1 Score: ', f1_score(res, ytest))
	# print('Precision Score: ', precision_score(res, ytest))
	# print('Recall Score: ', recall_score(res, ytest))
	# print('Confusion Matrix: ', confusion_matrix(res, ytest))

	# print('\n\n')
	# for i in range(50):
	# 	long = gj['features'][i]['properties']['Longitude']
	# 	lat = gj['features'][i]['properties']['Latitude']

	# 	trow, tcol = tempr.index(long, lat)
	# 	prow, pcol = prec.index(long, lat)
	# 	srow, scol = ghi.index(long, lat)

	# 	tval = tempdata[trow, tcol]
	# 	pval = precdata[prow, pcol]
	# 	sval = ghidata[srow, scol]

	# 	inp = np.array([tval, pval, sval])
	# 	inp = inp.reshape(1, len(inp))

	# 	res = model.predict(inp)
		
	# 	print('Lat: ', lat, ' Long: ',long,' Label: ',labels[i][0],' Feasibility: ', res[0][0]*100)
inter()
# test()