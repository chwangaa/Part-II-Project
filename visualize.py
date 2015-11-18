import csv

def parseCSV(file_name):
	with open(file_name, 'rb') as datafile:
		reader = csv.reader(datafile, delimiter=',')
		data = []
		for row in reader:
			entry = [int(row[0]), float(row[-1])]
			data.append(entry)
	return data

def getScatter(file_name):
	from math import log
	with open(file_name, 'rb') as datafile:
		reader = csv.reader(datafile, delimiter=',')
		data = [row for row in reader]
		xs = [int(row[0]) for row in data if float(row[-1]) > 0]
		ys = [log(float(row[-1]), 8) for row in data if float(row[-1]) > 0]
		return xs, ys

def plotScatter(file_names, title):
	import matplotlib.pyplot as plt
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	markers = [['b', 's'], ['r', 'o'], ['g', 'x'], ['y', '+']]
	i = 0
	for file_name in file_names:
		xs, ys = getScatter(file_name)
		ax1.scatter(xs, ys, c=markers[i][0], marker=markers[i][1], label=file_name)
		i+=1

	plt.legend(loc='lower right')
	plt.title(title)
	plt.show()

def normalize(data):
	'''the data must be a list of 2-tuple
	the return data makes sure the first element only appears once'''

	# use one dictionary to track the total accumulated time
	# use another to track the count
	from collections import defaultdict
	accum = defaultdict(int)
	count = defaultdict(int)

	for size, time in data:
		accum[size] += time
		count[size] += 1

	# iterate over the dictionary to calculate the average value
	averages = []
	for size, time in accum.iteritems():
		averages.append([size, time/count[size]])

	return averages