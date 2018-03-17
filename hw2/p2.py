import csv
import numpy as np

# Population 1b
pop = []
with open('population1b.csv','rb') as population:
	population_reader = csv.reader(population)
	pointer_loc = 0
	for row in population_reader:
		if pointer_loc > 0:
			pop.append(float(row[1]))
		pointer_loc += 1	

pop1_mean = np.mean(pop)
pop1_var = np.var(pop)
pop1_size = len(pop)

print 'Population1b has size: ', pop1_size, ' mean: ', pop1_mean, ' variance: ', pop1_var	

# Population 2b
pop = []
with open('population2b.csv','rb') as population:
	population_reader = csv.reader(population)
	pointer_loc = 0
	for row in population_reader:
		if pointer_loc > 0:
			pop.append(float(row[1]))
		pointer_loc += 1	

pop2_mean = np.mean(pop)
pop2_var = np.var(pop)
pop2_size = len(pop)

print 'Population2b has size: ', pop2_size, ' mean: ', pop2_mean, ' variance: ', pop2_var

# Population 3b
pop = []
with open('population3b.csv','rb') as population:
	population_reader = csv.reader(population)
	pointer_loc = 0
	for row in population_reader:
		if pointer_loc > 0:
			pop.append(float(row[1]))
		pointer_loc += 1	

pop3_mean = np.mean(pop)
pop3_var = np.var(pop)
pop3_size = len(pop)

print 'Population3b has size: ', pop3_size, ' mean: ', pop3_mean, ' variance: ', pop3_var

# Population 4b
pop = []
with open('population4b.csv','rb') as population:
	population_reader = csv.reader(population)
	pointer_loc = 0
	for row in population_reader:
		if pointer_loc > 0:
			pop.append(float(row[1]))
		pointer_loc += 1	

pop4_mean = np.mean(pop)
pop4_var = np.var(pop)
pop4_size = len(pop)

print 'Population4b has size: ', pop4_size, ' mean: ', pop4_mean, ' variance: ', pop4_var

# Population 1p
pop = []
with open('population1p.csv','rb') as population:
	population_reader = csv.reader(population)
	pointer_loc = 0
	for row in population_reader:
		if pointer_loc > 0:
			pop.append(float(row[1]))
		pointer_loc += 1	

pop1p_mean = np.mean(pop)
pop1p_var = np.var(pop)
pop1p_size = len(pop)

print 'Population1p has size: ', pop1_size, ' mean: ', pop1p_mean, ' variance: ', pop1p_var

# Population 2p
pop = []
with open('population2p.csv','rb') as population:
	population_reader = csv.reader(population)
	pointer_loc = 0
	for row in population_reader:
		if pointer_loc > 0:
			pop.append(float(row[1]))
		pointer_loc += 1	

pop2p_mean = np.mean(pop)
pop2p_var = np.var(pop)
pop2p_size = len(pop)

print 'Population2p has size: ', pop2p_size, ' mean: ', pop2p_mean, ' variance: ', pop2p_var

# Population 3p
pop = []
with open('population3p.csv','rb') as population:
	population_reader = csv.reader(population)
	pointer_loc = 0
	for row in population_reader:
		if pointer_loc > 0:
			pop.append(float(row[1]))
		pointer_loc += 1	

pop3p_mean = np.mean(pop)
pop3p_var = np.var(pop)
pop3p_size = len(pop)

print 'Population3p has size: ', pop3p_size, ' mean: ', pop3p_mean, ' variance: ', pop3p_var

# Population 4p
pop = []
with open('population4p.csv','rb') as population:
	population_reader = csv.reader(population)
	pointer_loc = 0
	for row in population_reader:
		if pointer_loc > 0:
			pop.append(float(row[1]))
		pointer_loc += 1	

pop4p_mean = np.mean(pop)
pop4p_var = np.var(pop)
pop4p_size = len(pop)

print 'Population4p has size: ', pop4p_size, ' mean: ', pop4p_mean, ' variance: ', pop4p_var

SE1b2b = (pop1_var/pop1_size + pop2_var/pop2_size)**(0.5)
td1b2b = (pop1_mean - pop2_mean) / SE1b2b
print 'The td score for 1b and 2b is: ', td1b2b 

SE3b4b = (pop3_var/pop3_size + pop4_var/pop4_size)**(0.5)
td3b4b = (pop3_mean - pop4_mean) / SE3b4b
print 'The td score for 3b and 4b is: ', td3b4b 

SE1p2p = (pop1p_var/pop1p_size + pop2p_var/pop2p_size)**(0.5)
td1p2p = (pop1p_mean - pop2p_mean) / SE1p2p
print 'The td score for 1p and 2p is: ', td1p2p 

SE3p4p = (pop3p_var/pop3p_size + pop4p_var/pop4p_size)**(0.5)
td3p4p = (pop3p_mean - pop4p_mean) / SE3p4p
print 'The td score for 3p and 4p is: ', td3p4p  

