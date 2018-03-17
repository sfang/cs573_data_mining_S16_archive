import csv
import numpy as np
import math
import random
import matplotlib.pyplot as plt

# Population 1b
pop1b = []
with open('population1b.csv','rb') as population:
	population_reader = csv.reader(population)
	pointer_loc = 0
	for row in population_reader:
		if pointer_loc > 0:
			if pointer_loc < 1001:
				pop1b.append(float(row[1]))
			else:
				break	
		pointer_loc += 1
		
# Population 2b
pop2b = []
with open('population2b.csv','rb') as population:
	population_reader = csv.reader(population)
	pointer_loc = 0
	for row in population_reader:
		if pointer_loc > 0:
			if pointer_loc < 1001:
				pop2b.append(float(row[1]))
			else:
				break	
		pointer_loc += 1

# Population 3b
pop3b = []
with open('population3b.csv','rb') as population:
	population_reader = csv.reader(population)
	pointer_loc = 0
	for row in population_reader:
		if pointer_loc > 0:
			if pointer_loc < 1001:
				pop3b.append(float(row[1]))
			else:
				break	
		pointer_loc += 1

# Population 1b
pop4b = []
with open('population4b.csv','rb') as population:
	population_reader = csv.reader(population)
	pointer_loc = 0
	for row in population_reader:
		if pointer_loc > 0:
			if pointer_loc < 1001:
				pop4b.append(float(row[1]))
			else:
				break	
		pointer_loc += 1

n1t = []
n2t = []
n3t = []
n4t = []

# initialization
n1t.append(0)
n2t.append(0)
n3t.append(0)
n4t.append(1)

n1t.append(0)
n2t.append(0)
n3t.append(1)
n4t.append(1)

n1t.append(0)
n2t.append(1)
n3t.append(1)
n4t.append(1)

n1t.append(1)
n2t.append(1)
n3t.append(1)
n4t.append(1)

reward_sequence = []
reward_sequence.append(1)

#print sum(pop4b[0:4])
for t in range(4,1000):
	lastn1 = n1t[t-1]
	lastn2 = n2t[t-1]
	lastn3 = n3t[t-1]
	lastn4 = n4t[t-1]
	tmp_n1 = lastn1 + 1
	tmp_n2 = lastn2 + 1
	tmp_n3 = lastn3 + 1
	tmp_n4 = lastn4 + 1
	#print tmp_n1, tmp_n2, tmp_n3, tmp_n4
	score_n1 = (1/(tmp_n1-1))*(sum(pop1b[0:lastn1])) + (2*math.log(t+1)/(tmp_n1-1))**(0.5)
	score_n2 = (1/(tmp_n2-1))*(sum(pop2b[0:lastn2])) + (2*math.log(t+1)/(tmp_n2-1))**(0.5)
	score_n3 = (1/(tmp_n3-1))*(sum(pop3b[0:lastn3])) + (2*math.log(t+1)/(tmp_n3-1))**(0.5)
	score_n4 = (1/(tmp_n4-1))*(sum(pop4b[0:lastn4])) + (2*math.log(t+1)/(tmp_n4-1))**(0.5)
	score_vec = [score_n1, score_n2, score_n3, score_n4]
	if score_n1 == max(score_vec):
		n1t.append(lastn1+1)
		n2t.append(lastn2)
		n3t.append(lastn3)
		n4t.append(lastn4)
		reward_sequence.append(pop1b[tmp_n1])
		print 'At time: ', t, ' play arm: 1', score_n1, max(score_vec)
	elif score_n2 == max(score_vec):
		n1t.append(lastn1)
		n2t.append(lastn2+1)
		n3t.append(lastn3)
		n4t.append(lastn4)
		reward_sequence.append(pop2b[tmp_n2])
		print 'At time: ', t, ' play arm: 2', score_n2, max(score_vec)
	elif score_n3 == max(score_vec):
		n1t.append(lastn1)
		n2t.append(lastn2)
		n3t.append(lastn3+1)
		n4t.append(lastn4)
		reward_sequence.append(pop3b[tmp_n3])
		print 'At time: ', t, ' play arm: 3', score_n3, max(score_vec)
	else:
		n1t.append(lastn1)
		n2t.append(lastn2)
		n3t.append(lastn3)
		n4t.append(lastn4+1)
		reward_sequence.append(pop4b[tmp_n4])
		print 'At time: ', t, ' play arm: 4', score_n4, max(score_vec)
	del score_vec

#Rt_vec = [sum(pop1b[0:max(n1t)]), sum(pop2b[0:max(n2t)]), sum(pop3b[0:max(n3t)]), sum(pop4b[0:max(n4t)])]
Rt = sum(reward_sequence)
print 'The total reward for ib: ', Rt

ib_UCB1_mean = np.mean(reward_sequence)
ib_UCBI_var = np.var(reward_sequence)
ib_UCBI_len = len(reward_sequence)

plt.plot(n1t, color = 'red')
plt.hold(True)
plt.plot(n2t, color = 'green')
plt.plot(n3t, color = 'blue')
plt.plot(n4t, color = 'black')
plt.title('The n_i(t) for each arm, ib population, UCBI')
plt.ylabel('The n_i(t) for each arm i pulled')
plt.xlabel('The time t (each arm: red-1b, green-2b, blue-3b, black-4b)')
plt.show()


# Population 1p
pop1p = []
with open('population1p.csv','rb') as population:
	population_reader = csv.reader(population)
	pointer_loc = 0
	for row in population_reader:
		if pointer_loc > 0:
			if pointer_loc < 1001:
				pop1p.append(float(row[1]))
			else:
				break	
		pointer_loc += 1
		
# Population 2p
pop2p = []
with open('population2p.csv','rb') as population:
	population_reader = csv.reader(population)
	pointer_loc = 0
	for row in population_reader:
		if pointer_loc > 0:
			if pointer_loc < 1001:
				pop2p.append(float(row[1]))
			else:
				break	
		pointer_loc += 1

# Population 3p
pop3p = []
with open('population3p.csv','rb') as population:
	population_reader = csv.reader(population)
	pointer_loc = 0
	for row in population_reader:
		if pointer_loc > 0:
			if pointer_loc < 1001:
				pop3p.append(float(row[1]))
			else:
				break	
		pointer_loc += 1

# Population 4p
pop4p = []
with open('population4p.csv','rb') as population:
	population_reader = csv.reader(population)
	pointer_loc = 0
	for row in population_reader:
		if pointer_loc > 0:
			if pointer_loc < 1001:
				pop4p.append(float(row[1]))
			else:
				break	
		pointer_loc += 1

del n1t, n2t, n3t, n4t

n1t = []
n2t = []
n3t = []
n4t = []
Rt = 0
reward_sequence = []
# initialization
n1t.append(0)
n2t.append(0)
n3t.append(0)
n4t.append(1)

n1t.append(0)
n2t.append(0)
n3t.append(1)
n4t.append(1)

n1t.append(0)
n2t.append(1)
n3t.append(1)
n4t.append(1)

n1t.append(1)
n2t.append(1)
n3t.append(1)
n4t.append(1)

for t in range(4,1000):
	lastn1 = n1t[t-1]
	lastn2 = n2t[t-1]
	lastn3 = n3t[t-1]
	lastn4 = n4t[t-1]
	tmp_n1 = lastn1 + 1
	tmp_n2 = lastn2 + 1
	tmp_n3 = lastn3 + 1
	tmp_n4 = lastn4 + 1
	print tmp_n1, tmp_n2, tmp_n3, tmp_n4
	score_n1 = (1/(tmp_n1-1))*(sum(pop1p[0:tmp_n1-1])) + (2*math.log(t+1)/(tmp_n1-1))**(0.5)
	score_n2 = (1/(tmp_n2-1))*(sum(pop2p[0:tmp_n2-1])) + (2*math.log(t+1)/(tmp_n2-1))**(0.5)
	score_n3 = (1/(tmp_n3-1))*(sum(pop3p[0:tmp_n3-1])) + (2*math.log(t+1)/(tmp_n3-1))**(0.5)
	score_n4 = (1/(tmp_n4-1))*(sum(pop4p[0:tmp_n4-1])) + (2*math.log(t+1)/(tmp_n4-1))**(0.5)	
	# score_n1 = (1/(tmp_n1))*(sum(reward_sequence[0:tmp_n1])) + (2*math.log(t+1)/tmp_n1)**(0.5)
	# score_n2 = (1/(tmp_n2))*(sum(reward_sequence[0:tmp_n2])) + (2*math.log(t+1)/tmp_n2)**(0.5)
	# score_n3 = (1/(tmp_n3))*(sum(reward_sequence[0:tmp_n3])) + (2*math.log(t+1)/tmp_n3)**(0.5)
	# score_n4 = (1/(tmp_n4))*(sum(reward_sequence[0:tmp_n4])) + (2*math.log(t+1)/tmp_n4)**(0.5)	
	# score_n1 = (1/(tmp_n1))*(sum(pop1p[0:tmp_n1])) + ((2*math.log(t+1))/tmp_n1)**(0.5)
	# score_n2 = (1/(tmp_n2))*(sum(pop2p[0:tmp_n2])) + ((2*math.log(t+1))/tmp_n2)**(0.5)
	# score_n3 = (1/(tmp_n3))*(sum(pop3p[0:tmp_n3])) + ((2*math.log(t+1))/tmp_n3)**(0.5)
	# score_n4 = (1/(tmp_n4))*(sum(pop4p[0:tmp_n4])) + ((2*math.log(t+1))/tmp_n4)**(0.5)
	score_vec = [score_n1, score_n2, score_n3, score_n4]
	if score_n1 == max(score_vec):
		n1t.append(lastn1+1)
		n2t.append(lastn2)
		n3t.append(lastn3)
		n4t.append(lastn4)
		reward_sequence.append(pop1p[tmp_n1])
		print 'At time: ', t, ' play arm: 1', score_n1, max(score_vec)
		#print 'At time: ', t, ' play arm: 1', score_n1, score_vec
	elif score_n2 == max(score_vec):
		n1t.append(lastn1)
		n2t.append(lastn2+1)
		n3t.append(lastn3)
		n4t.append(lastn4)
		reward_sequence.append(pop2p[tmp_n2])
		print 'At time: ', t, ' play arm: 2', score_n2, max(score_vec)
	elif score_n3 == max(score_vec):
		n1t.append(lastn1)
		n2t.append(lastn2)
		n3t.append(lastn3+1)
		n4t.append(lastn4)
		reward_sequence.append(pop3p[tmp_n3])
		print 'At time: ', t, ' play arm: 3', score_n3, max(score_vec)
	else:
		n1t.append(lastn1)
		n2t.append(lastn2)
		n3t.append(lastn3)
		n4t.append(lastn4+1)
		reward_sequence.append(pop4p[tmp_n4])
		print 'At time: ', t, ' play arm: 4', score_n4, max(score_vec)

#Rt_vec = [sum(pop1p[0:max(n1t)]), sum(pop2p[0:max(n2t)]), sum(pop3p[0:max(n3t)]), sum(pop4p[0:max(n4t)])]
Rt = sum(reward_sequence)
print 'The total reward for ib: ', Rt

ip_UCBI_mean = np.mean(reward_sequence)
ip_UCBI_var = np.var(reward_sequence)
ip_UCBI_len = len(reward_sequence)

plt.plot(n1t, color = 'red')
plt.hold(True)
plt.plot(n2t, color = 'green')
plt.plot(n3t, color = 'blue')
plt.plot(n4t, color = 'black')
plt.title('The n_i(t) for each arm, ip population, UCBI')
plt.ylabel('The n_i(t) for each arm i pulled')
plt.xlabel('The time t (each arm: red-1p, green-2p, blue-3p, black-4p)')
plt.show()

# epsilon greedy for ib
del n1t, n2t, n3t, n4t
n1t = []
n2t = []
n3t = []
n4t = []
Rt = 0
reward_sequence = []
# initialization
n1t.append(0)
n2t.append(0)
n3t.append(0)
n4t.append(1)

n1t.append(0)
n2t.append(0)
n3t.append(1)
n4t.append(1)

n1t.append(0)
n2t.append(1)
n3t.append(1)
n4t.append(1)

n1t.append(1)
n2t.append(1)
n3t.append(1)
n4t.append(1)

for t in range(4,1000):
	lastn1 = n1t[t-1]
	lastn2 = n2t[t-1]
	lastn3 = n3t[t-1]
	lastn4 = n4t[t-1]
	mu1 = sum(pop1b[0:lastn1])/(lastn1)
	mu2 = sum(pop2b[0:lastn2])/(lastn2)
	mu3 = sum(pop3b[0:lastn3])/(lastn3)
	mu4 = sum(pop4b[0:lastn4])/(lastn4)
	mu_vec = [mu1,mu2,mu3,mu4]
	mu_star = max(mu_vec)
	#print mu_vec, mu_star
	delta_vec = []
	for mu in mu_vec:
		delta_vec.append(-mu+mu_star)		
	max_delta = max(delta_vec)	
	min_delta = max_delta

	opt_cnt = 0
	for delta in delta_vec:
		if delta > 0:
			if delta <= min_delta:
				min_delta = delta
		if delta == 0:
			opt_ind = opt_cnt
		opt_cnt += 1			

	print 'Delta min value is: ', min_delta		

	epsilon = min([12/(t*min_delta**2), 1])	
	randnum = random.uniform(0,1)

	if randnum < (1 - epsilon):
		# Play the best suboptimal
		if opt_ind == 1:
	 		n1t.append(lastn1+1)
	 		n2t.append(lastn2)
	 		n3t.append(lastn3)
	 		n4t.append(lastn4)
	 		Rt = Rt + pop1b[lastn1]
	 		reward_sequence.append(pop1b[lastn1])
	 	elif opt_ind == 2:	
	 		n1t.append(lastn1)
	 		n2t.append(lastn2+1)
	 		n3t.append(lastn3)
	 		n4t.append(lastn4)
	 		Rt = Rt + pop2b[lastn2]
	 		reward_sequence.append(pop2b[lastn2])
	 	elif opt_ind == 3:
	 		n1t.append(lastn1)
	 		n2t.append(lastn2)
	 		n3t.append(lastn3+1)
	 		n4t.append(lastn4)
	 		Rt = Rt + pop3b[lastn3]	 
	 		reward_sequence.append(pop3b[lastn3])
	 	else:
	 		n1t.append(lastn1)
	 		n2t.append(lastn2)
	 		n3t.append(lastn3)
	 		n4t.append(lastn4+1)
	 		Rt = Rt + pop4b[lastn4]
	 		reward_sequence.append(pop4b[lastn4])
	else:
		rand_ind = random.randint(1, 4)
		if rand_ind == 1:
	 		n1t.append(lastn1+1)
	 		n2t.append(lastn2)
	 		n3t.append(lastn3)
	 		n4t.append(lastn4)
	 		Rt = Rt + pop1b[lastn1]
	 		reward_sequence.append(pop1b[lastn1])
	 	elif rand_ind == 2:	
	 		n1t.append(lastn1)
	 		n2t.append(lastn2+1)
	 		n3t.append(lastn3)
	 		n4t.append(lastn4)
	 		Rt = Rt + pop2b[lastn2]
	 		reward_sequence.append(pop2b[lastn2])
	 	elif rand_ind == 3:
	 		n1t.append(lastn1)
	 		n2t.append(lastn2)
	 		n3t.append(lastn3+1)
	 		n4t.append(lastn4)
	 		Rt = Rt + pop3b[lastn3]	 
	 		reward_sequence.append(pop3b[lastn3])
	 	else:
	 		n1t.append(lastn1)
	 		n2t.append(lastn2)
	 		n3t.append(lastn3)
	 		n4t.append(lastn4+1)
	 		Rt = Rt + pop4b[lastn4]
	 		reward_sequence.append(pop4b[lastn4])
				
	#score_vec = [score_n1, score_n2, score_n3, score_n4]
	# if score_n1 == max(score_vec):
	# 	n1t.append(lastn1+1)
	# 	n2t.append(lastn2)
	# 	n3t.append(lastn3)
	# 	n4t.append(lastn4)
	# 	Rt = Rt + pop1b[t]
	# 	print 'At time: ', t, ' play arm: 1', score_n1, max(score_vec)
	# 	#print 'At time: ', t, ' play arm: 1', score_n1, score_vec
	# elif score_n2 == max(score_vec):
	# 	n1t.append(lastn1)
	# 	n2t.append(lastn2+1)
	# 	n3t.append(lastn3)
	# 	n4t.append(lastn4)
	# 	Rt = Rt + pop2b[t]
	# 	print 'At time: ', t, ' play arm: 2', score_n2, max(score_vec)
	# elif score_n3 == max(score_vec):
	# 	n1t.append(lastn1)
	# 	n2t.append(lastn2)
	# 	n3t.append(lastn3+1)
	# 	n4t.append(lastn4)
	# 	Rt = Rt + pop3b[t]
	# 	print 'At time: ', t, ' play arm: 3', score_n3, max(score_vec)
	# else:
	# 	n1t.append(lastn1)
	# 	n2t.append(lastn2)
	# 	n3t.append(lastn3)
	# 	n4t.append(lastn4+1)
	# 	Rt = Rt + pop4b[t]
	# 	print 'At time: ', t, ' play arm: 4', score_n4, max(score_vec)

print 'The total reward for ib: ', Rt

ib_greedy_mean = np.mean(reward_sequence)
ib_greedy_var = np.var(reward_sequence)
ib_greedy_len = len(reward_sequence)

plt.plot(n1t, color = 'red')
plt.hold(True)
plt.plot(n2t, color = 'green')
plt.plot(n3t, color = 'blue')
plt.plot(n4t, color = 'black')
plt.title('The n_i(t) for each arm, ib population, epsilon-greedy')
plt.ylabel('The n_i(t) for each arm i pulled')
plt.xlabel('The time t (each arm: red-1b, green-2b, blue-3b, black-4b)')
plt.show()

# epsilon greedy for ip
del n1t, n2t, n3t, n4t
n1t = []
n2t = []
n3t = []
n4t = []
Rt = 0
reward_sequence = []
# initialization
n1t.append(0)
n2t.append(0)
n3t.append(1)
n4t.append(0)

n1t.append(1)
n2t.append(0)
n3t.append(1)
n4t.append(0)

n1t.append(1)
n2t.append(1)
n3t.append(1)
n4t.append(0)

n1t.append(1)
n2t.append(1)
n3t.append(1)
n4t.append(1)

for t in range(4,1000):
	lastn1 = n1t[t-1]
	lastn2 = n2t[t-1]
	lastn3 = n3t[t-1]
	lastn4 = n4t[t-1]
	mu1 = sum(pop1p[0:lastn1])/(lastn1)
	mu2 = sum(pop2p[0:lastn2])/(lastn2)
	mu3 = sum(pop3p[0:lastn3])/(lastn3)
	mu4 = sum(pop4p[0:lastn4])/(lastn4)
	mu_vec = [mu1,mu2,mu3,mu4]
	mu_star = max(mu_vec)
	#print mu_vec, mu_star
	delta_vec = []
	for mu in mu_vec:
		delta_vec.append(-mu+mu_star)		
	max_delta = max(delta_vec)	
	min_delta = max_delta

	opt_cnt = 0
	for delta in delta_vec:
		if delta > 0:
			if delta <= min_delta:
				min_delta = delta
		if delta == 0:
			opt_ind = opt_cnt
		opt_cnt += 1			

	print 'Delta min value is: ', min_delta		

	epsilon = min([12/(t*min_delta**2), 1])	
	randnum = random.uniform(0,1)

	if randnum < (1 - epsilon):
		# Play the best suboptimal
		if opt_ind == 1:
	 		n1t.append(lastn1+1)
	 		n2t.append(lastn2)
	 		n3t.append(lastn3)
	 		n4t.append(lastn4)
	 		Rt = Rt + pop1p[lastn1]
	 		reward_sequence.append(pop1p[lastn1])
	 	elif opt_ind == 2:	
	 		n1t.append(lastn1)
	 		n2t.append(lastn2+1)
	 		n3t.append(lastn3)
	 		n4t.append(lastn4)
	 		Rt = Rt + pop2p[lastn2]
	 		reward_sequence.append(pop2p[lastn2])
	 	elif opt_ind == 3:
	 		n1t.append(lastn1)
	 		n2t.append(lastn2)
	 		n3t.append(lastn3+1)
	 		n4t.append(lastn4)
	 		Rt = Rt + pop3p[lastn3]	 
	 		reward_sequence.append(pop3p[lastn3])
	 	else:
	 		n1t.append(lastn1)
	 		n2t.append(lastn2)
	 		n3t.append(lastn3)
	 		n4t.append(lastn4+1)
	 		Rt = Rt + pop4p[lastn4]
	 		reward_sequence.append(pop4p[lastn4])
	else:
		rand_ind = random.randint(1, 4)
		if rand_ind == 1:
	 		n1t.append(lastn1+1)
	 		n2t.append(lastn2)
	 		n3t.append(lastn3)
	 		n4t.append(lastn4)
	 		Rt = Rt + pop1p[lastn1]
	 		reward_sequence.append(pop1p[lastn1])
	 	elif rand_ind == 2:	
	 		n1t.append(lastn1)
	 		n2t.append(lastn2+1)
	 		n3t.append(lastn3)
	 		n4t.append(lastn4)
	 		Rt = Rt + pop2p[lastn2]
	 		reward_sequence.append(pop2p[lastn2])
	 	elif rand_ind == 3:
	 		n1t.append(lastn1)
	 		n2t.append(lastn2)
	 		n3t.append(lastn3+1)
	 		n4t.append(lastn4)
	 		Rt = Rt + pop3p[lastn3]
	 		reward_sequence.append(pop3p[lastn3])	 
	 	else:
	 		n1t.append(lastn1)
	 		n2t.append(lastn2)
	 		n3t.append(lastn3)
	 		n4t.append(lastn4+1)
	 		Rt = Rt + pop4p[lastn4]	
	 		reward_sequence.append(pop4p[lastn4])
	# lastn1 = n1t[t-1]
	# lastn2 = n2t[t-1]
	# lastn3 = n3t[t-1]
	# lastn4 = n4t[t-1]
	# score_n1 = sum(pop1p[0:t+1])/(t+1)
	# score_n2 = sum(pop2p[0:t+1])/(t+1)
	# score_n3 = sum(pop3p[0:t+1])/(t+1)
	# score_n4 = sum(pop4p[0:t+1])/(t+1)
	# score_vec = [score_n1, score_n2, score_n3, score_n4]
	# if score_n1 == max(score_vec):
	# 	n1t.append(lastn1+1)
	# 	n2t.append(lastn2)
	# 	n3t.append(lastn3)
	# 	n4t.append(lastn4)
	# 	Rt = Rt + pop1p[t]
	# 	print 'At time: ', t, ' play arm: 1', score_n1, max(score_vec)
	# 	#print 'At time: ', t, ' play arm: 1', score_n1, score_vec
	# elif score_n2 == max(score_vec):
	# 	n1t.append(lastn1)
	# 	n2t.append(lastn2+1)
	# 	n3t.append(lastn3)
	# 	n4t.append(lastn4)
	# 	Rt = Rt + pop2p[t]
	# 	print 'At time: ', t, ' play arm: 2', score_n2, max(score_vec)
	# elif score_n3 == max(score_vec):
	# 	n1t.append(lastn1)
	# 	n2t.append(lastn2)
	# 	n3t.append(lastn3+1)
	# 	n4t.append(lastn4)
	# 	Rt = Rt + pop3p[t]
	# 	print 'At time: ', t, ' play arm: 3', score_n3, max(score_vec)
	# else:
	# 	n1t.append(lastn1)
	# 	n2t.append(lastn2)
	# 	n3t.append(lastn3)
	# 	n4t.append(lastn4+1)
	# 	Rt = Rt + pop4p[t]
	# 	print 'At time: ', t, ' play arm: 4', score_n4, max(score_vec)

print 'The total reward for ib: ', Rt

ip_greedy_mean = np.mean(reward_sequence)
ip_greedy_var = np.var(reward_sequence)
ip_greedy_len = len(reward_sequence)

plt.plot(n1t, color = 'red')
plt.hold(True)
plt.plot(n2t, color = 'green')
plt.plot(n3t, color = 'blue')
plt.plot(n4t, color = 'black')
plt.title('The n_i(t) for each arm, ip population, epsilon-greedy')
plt.ylabel('The n_i(t) for each arm i pulled')
plt.xlabel('The time t (each arm: red-1p, green-2p, blue-3p, black-4p)')
plt.show()

SE_b = (ib_greedy_var/ib_greedy_len + ib_UCBI_var/ib_UCBI_len)**(0.5)
td_b = (ib_greedy_mean - ib_UCB1_mean)/SE_b
print 'td ib is:', td_b

print ip_greedy_mean, ip_UCBI_mean
print ip_greedy_len, ip_UCBI_len
SE_p = (ip_greedy_var/ip_greedy_len + ip_UCBI_var/ip_UCBI_len)**(0.5)
td_p = (ip_greedy_mean - ip_UCBI_mean)/SE_p
print 'td ip is:', td_p


# Thompson sampling
n1t = []
n2t = []
n3t = []
n4t = []
n1t.append(0)
n2t.append(0)
n3t.append(0)
n4t.append(0)
S1 = 0
F1 = 0
S2 = 0
F2 = 0
S3 = 0
F3 = 0
S4 = 0
F4 = 0
reward_sequence = []
for t in range(0,1000):
	mu1 = np.random.beta(1+S1, 1+F1)
	mu2 = np.random.beta(1+S2, 1+F2)
	mu3 = np.random.beta(1+S3, 1+F3)
	mu4 = np.random.beta(1+S4, 1+F4)
	lastn1 = n1t[t]
	lastn2 = n2t[t]
	lastn3 = n3t[t]
	lastn4 = n4t[t]
	mu_vec = [mu1, mu2, mu3, mu4]
	if mu1 == max(mu_vec):
		n1t.append(lastn1+1)
		n2t.append(lastn2)
		n3t.append(lastn3)
		n4t.append(lastn4)
		reward_sequence.append(pop1b[lastn1])
		S1 = S1 + pop1b[lastn1]
		F1 = F1 + (1 - pop1b[lastn1])
	elif mu2 == max(mu_vec):
		n1t.append(lastn1)
		n2t.append(lastn2+1)
		n3t.append(lastn3)
		n4t.append(lastn4)
		reward_sequence.append(pop2b[lastn2])
		S2 = S2 + pop2b[lastn2]
		F2 = F2 + (1 - pop2b[lastn2])
	elif mu3 == max(mu_vec):
		n1t.append(lastn1)
		n2t.append(lastn2)
		n3t.append(lastn3+1)
		n4t.append(lastn4)
		reward_sequence.append(pop3b[lastn3])
		S3 = S3 + pop3b[lastn3]
		F3 = F3 + (1 - pop3b[lastn3])
	else:
		n1t.append(lastn1)
		n2t.append(lastn2)
		n3t.append(lastn3)
		n4t.append(lastn4+1)
		reward_sequence.append(pop4b[lastn4])
		S4 = S4 + pop4b[lastn4]
		F4 = F4 + (1 - pop4b[lastn4])	

print 'Final reward: ', sum(reward_sequence)			
plt.plot(n1t, color = 'red')
plt.hold(True)
plt.plot(n2t, color = 'green')
plt.plot(n3t, color = 'blue')
plt.plot(n4t, color = 'black')
plt.title('The n_i(t) for each arm, ib population, Thompson sampling, Beta(1,1)')
plt.ylabel('The n_i(t) for each arm i pulled')
plt.xlabel('The time t (each arm: red-1b, green-2b, blue-3b, black-4b)')
plt.show()

# Thompson sampling
n1t = []
n2t = []
n3t = []
n4t = []
n1t.append(0)
n2t.append(0)
n3t.append(0)
n4t.append(0)
S1 = 0
F1 = 0
S2 = 0
F2 = 0
S3 = 0
F3 = 0
S4 = 0
F4 = 0
reward_sequence = []
for t in range(0,1000):
	mu1 = np.random.beta(100+S1, 100+F1)
	mu2 = np.random.beta(100+S2, 100+F2)
	mu3 = np.random.beta(100+S3, 100+F3)
	mu4 = np.random.beta(100+S4, 100+F4)
	lastn1 = n1t[t]
	lastn2 = n2t[t]
	lastn3 = n3t[t]
	lastn4 = n4t[t]
	mu_vec = [mu1, mu2, mu3, mu4]
	if mu1 == max(mu_vec):
		n1t.append(lastn1+1)
		n2t.append(lastn2)
		n3t.append(lastn3)
		n4t.append(lastn4)
		reward_sequence.append(pop1b[lastn1])
		S1 = S1 + pop1b[lastn1]
		F1 = F1 + (1 - pop1b[lastn1])
	elif mu2 == max(mu_vec):
		n1t.append(lastn1)
		n2t.append(lastn2+1)
		n3t.append(lastn3)
		n4t.append(lastn4)
		reward_sequence.append(pop2b[lastn2])
		S2 = S2 + pop2b[lastn2]
		F2 = F2 + (1 - pop2b[lastn2])
	elif mu3 == max(mu_vec):
		n1t.append(lastn1)
		n2t.append(lastn2)
		n3t.append(lastn3+1)
		n4t.append(lastn4)
		reward_sequence.append(pop3b[lastn3])
		S3 = S3 + pop3b[lastn3]
		F3 = F3 + (1 - pop3b[lastn3])
	else:
		n1t.append(lastn1)
		n2t.append(lastn2)
		n3t.append(lastn3)
		n4t.append(lastn4+1)
		reward_sequence.append(pop4b[lastn4])
		S4 = S4 + pop4b[lastn4]
		F4 = F4 + (1 - pop4b[lastn4])	

print 'Final reward: ', sum(reward_sequence), 'length: ', len(reward_sequence)			
plt.plot(n1t, color = 'red')
plt.hold(True)
plt.plot(n2t, color = 'green')
plt.plot(n3t, color = 'blue')
plt.plot(n4t, color = 'black')
plt.title('The n_i(t) for each arm, ib population, Thompson sampling, Beta(100,100)')
plt.ylabel('The n_i(t) for each arm i pulled')
plt.xlabel('The time t (each arm: red-1b, green-2b, blue-3b, black-4b)')
plt.show()

# Thompson sampling
n1t = []
n2t = []
n3t = []
n4t = []
n1t.append(0)
n2t.append(0)
n3t.append(0)
n4t.append(0)
S1 = 0
F1 = 0
S2 = 0
F2 = 0
S3 = 0
F3 = 0
S4 = 0
F4 = 0
reward_sequence = []
for t in range(0,1000):
	mu1 = np.random.beta(0.0000000001+S1, 0.0000000001+F1)
	mu2 = np.random.beta(0.0000000001+S2, 0.0000000001+F2)
	mu3 = np.random.beta(0.0000000001+S3, 0.0000000001+F3)
	mu4 = np.random.beta(0.0000000001+S4, 0.0000000001+F4)
	lastn1 = n1t[t]
	lastn2 = n2t[t]
	lastn3 = n3t[t]
	lastn4 = n4t[t]
	mu_vec = [mu1, mu2, mu3, mu4]
	if mu1 == max(mu_vec):
		n1t.append(lastn1+1)
		n2t.append(lastn2)
		n3t.append(lastn3)
		n4t.append(lastn4)
		reward_sequence.append(pop1b[lastn1])
		S1 = S1 + pop1b[lastn1]
		F1 = F1 + (1 - pop1b[lastn1])
	elif mu2 == max(mu_vec):
		n1t.append(lastn1)
		n2t.append(lastn2+1)
		n3t.append(lastn3)
		n4t.append(lastn4)
		reward_sequence.append(pop2b[lastn2])
		S2 = S2 + pop2b[lastn2]
		F2 = F2 + (1 - pop2b[lastn2])
	elif mu3 == max(mu_vec):
		n1t.append(lastn1)
		n2t.append(lastn2)
		n3t.append(lastn3+1)
		n4t.append(lastn4)
		reward_sequence.append(pop3b[lastn3])
		S3 = S3 + pop3b[lastn3]
		F3 = F3 + (1 - pop3b[lastn3])
	else:
		n1t.append(lastn1)
		n2t.append(lastn2)
		n3t.append(lastn3)
		n4t.append(lastn4+1)
		reward_sequence.append(pop4b[lastn4])
		S4 = S4 + pop4b[lastn4]
		F4 = F4 + (1 - pop4b[lastn4])	

print 'Final reward: ', sum(reward_sequence)			
plt.plot(n1t, color = 'red')
plt.hold(True)
plt.plot(n2t, color = 'green')
plt.plot(n3t, color = 'blue')
plt.plot(n4t, color = 'black')
plt.title('The n_i(t) for each arm, ib population, Thompson sampling, Beta(0,0)')
plt.ylabel('The n_i(t) for each arm i pulled')
plt.xlabel('The time t (each arm: red-1b, green-2b, blue-3b, black-4b)')
plt.show()

# # Thompson sampling
# n1t = []
# n2t = []
# n3t = []
# n4t = []
# n1t.append(0)
# n2t.append(0)
# n3t.append(0)
# n4t.append(0)
# S1 = 0
# F1 = 0
# S2 = 0
# F2 = 0
# S3 = 0
# F3 = 0
# S4 = 0
# F4 = 0
# reward_sequence = []
# for t in range(0,1000):
# 	mu1 = np.random.beta(100+S1, 100+F1)
# 	mu2 = np.random.beta(100+S2, 100+F2)
# 	mu3 = np.random.beta(100+S3, 100+F3)
# 	mu4 = np.random.beta(100+S4, 100+F4)
# 	lastn1 = n1t[t]
# 	lastn2 = n2t[t]
# 	lastn3 = n3t[t]
# 	lastn4 = n4t[t]
# 	mu_vec = [mu1, mu2, mu3, mu4]
# 	if mu1 == max(mu_vec):
# 		n1t.append(lastn1+1)
# 		n2t.append(lastn2)
# 		n3t.append(lastn3)
# 		n4t.append(lastn4)
# 		reward_sequence.append(pop1p[lastn1])
# 		S1 = S1 + pop1p[lastn1]
# 		F1 = F1 + (1 - pop1p[lastn1])
# 	elif mu2 == max(mu_vec):
# 		n1t.append(lastn1)
# 		n2t.append(lastn2+1)
# 		n3t.append(lastn3)
# 		n4t.append(lastn4)
# 		reward_sequence.append(pop2p[lastn2])
# 		S2 = S2 + pop2p[lastn2]
# 		F2 = F2 + (1 - pop2p[lastn2])
# 	elif mu3 == max(mu_vec):
# 		n1t.append(lastn1)
# 		n2t.append(lastn2)
# 		n3t.append(lastn3+1)
# 		n4t.append(lastn4)
# 		reward_sequence.append(pop3p[lastn3])
# 		S3 = S3 + pop3p[lastn3]
# 		F3 = F3 + (1 - pop3p[lastn3])
# 	else:
# 		n1t.append(lastn1)
# 		n2t.append(lastn2)
# 		n3t.append(lastn3)
# 		n4t.append(lastn4+1)
# 		reward_sequence.append(pop4p[lastn4])
# 		S4 = S4 + pop4p[lastn4]
# 		F4 = F4 + (1 - pop4p[lastn4])	

# print 'Final reward: ', sum(reward_sequence), 'length: ', len(reward_sequence)			
# plt.plot(n1t, color = 'red')
# plt.hold(True)
# plt.plot(n2t, color = 'green')
# plt.plot(n3t, color = 'blue')
# plt.plot(n4t, color = 'black')
# plt.title('The n_i(t) for each arm, ib population, Thompson sampling, Beta(100,100)')
# plt.ylabel('The n_i(t) for each arm i pulled')
# plt.xlabel('The time t (each arm: red-1b, green-2b, blue-3b, black-4b)')
# plt.show()
