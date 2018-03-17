import csv
import numpy as np

with open('DEM_donations_2012.csv','rb') as DEM_file:
	DEM_reader = csv.reader(DEM_file)
	DEM_cnt = 0
	DEM_sum = 0
	DEM_S = 0
	pointer_loc = 0
	for row in DEM_reader:
		if pointer_loc > 0:
			a_donation = float(row[1])
			if a_donation > 0:
				DEM_sum = DEM_sum + a_donation
				DEM_S = DEM_S + (a_donation - 1980.79096793)**2
				DEM_cnt += 1
				#print float(row[1])
		pointer_loc += 1		
		#print row[1]

	DEM_mean = float(DEM_sum) / float(DEM_cnt)
	print 'DEM: Without Considering the negatives:', DEM_mean	#DEM_cnt += DEM_cnt

DEM_S = DEM_S / DEM_cnt
print DEM_S

# Also consider the nagatives
# with open('DEM_donations_2012.csv','rb') as DEM_file:
# 	DEM_reader = csv.reader(DEM_file)
# 	DEM_cnt = 0
# 	DEM_sum = 0
# 	pointer_loc = 0
# 	for row in DEM_reader:
# 		if pointer_loc > 0:
# 			a_donation = float(row[1])
# 			DEM_sum = DEM_sum + a_donation
# 			DEM_cnt += 1
# 				#print float(row[1])
# 		pointer_loc += 1		
# 		#print row[1]

# 	DEM_mean = float(DEM_sum) / float(DEM_cnt)
# 	print 'DEM: Also Considering the negatives: ', DEM_mean	#DEM_cnt += DEM_cnt

with open('GOP_donations_2012.csv','rb') as GOP_file:
	GOP_reader = csv.reader(GOP_file)
	GOP_cnt = 0
	GOP_sum = 0
	pointer_loc = 0
	GOP_S = 0
	for row in GOP_reader:
		if pointer_loc > 0:
			a_donation = float(row[1])
			if a_donation > 0:
				GOP_sum = GOP_sum + a_donation
				GOP_S = GOP_S + (a_donation - 2150.65378289)**2
				GOP_cnt += 1
				#print float(row[1])
		pointer_loc += 1		
		#print row[1]

	GOP_mean = float(GOP_sum) / float(GOP_cnt)
	print 'GOP: Without Considering the negatives:', GOP_mean	#DEM_cnt += DEM_cnt

GOP_S = GOP_S/GOP_cnt
print GOP_S

print 'The GOP variance: ', GOP_S
print 'The DEM variance: ', DEM_S
SE = ( GOP_S/GOP_cnt + DEM_S/DEM_cnt)**(0.5)
print 'The empirical standard error is:', SE
td = ((GOP_mean - DEM_mean) - 0)/SE
print 'The test statistica td is: ', td

# Also consider the nagatives
# with open('GOP_donations_2012.csv','rb') as GOP_file:
# 	GOP_reader = csv.reader(GOP_file)
# 	GOP_cnt = 0
# 	GOP_sum = 0
# 	pointer_loc = 0
# 	for row in GOP_reader:
# 		if pointer_loc > 0:
# 			a_donation = float(row[1])
# 			GOP_sum = GOP_sum + a_donation
# 			GOP_cnt += 1
# 				#print float(row[1])
# 		pointer_loc += 1		
# 		#print row[1]

# 	GOP_mean = float(GOP_sum) / float(GOP_cnt)
# 	print 'GOP: Also Considering the negatives: ', GOP_mean	#DEM_cnt += DEM_cnt

# Create GOP state list
with open('GOP_donations_2012.csv','rb') as GOP_file:
	GOP_reader = csv.reader(GOP_file)
	pointer_loc = 0
	GOP_state_list = []
	for row in GOP_reader:
		if pointer_loc > 0:
			tmp_state = str(row[2])
			if tmp_state in GOP_state_list:
				tmp_state
			else:
				GOP_state_list.append(tmp_state)
			# if pointer_loc < 2:
			# 	tmp_state = str(row[2])
			# 	old_state = tmp_state
			# 	state_list.append(old_state)
			# 	print old_state
			# else:
			# 	tmp_state = str(row[2])
			# 	if tmp_state == old_state:
			# 		tmp_state
			# 	else:
			# 		old_state = tmp_state
			# 		state_list.append(old_state)

			#tmp_state = str(row[2])		
			#print a_donation	
		pointer_loc += 1 

print GOP_state_list, len(GOP_state_list)

# Create DEM state list
with open('DEM_donations_2012.csv','rb') as DEM_file:
	DEM_reader = csv.reader(DEM_file)
	pointer_loc = 0
	DEM_state_list = []
	for row in DEM_reader:
		if pointer_loc > 0:
			tmp_state = str(row[2])
			if tmp_state in DEM_state_list:
				tmp_state
			else:
				DEM_state_list.append(tmp_state)
			# if pointer_loc < 2:
			# 	tmp_state = str(row[2])
			# 	old_state = tmp_state
			# 	state_list.append(old_state)
			# 	print old_state
			# else:
			# 	tmp_state = str(row[2])
			# 	if tmp_state == old_state:
			# 		tmp_state
			# 	else:
			# 		old_state = tmp_state
			# 		state_list.append(old_state)

			#tmp_state = str(row[2])		
			#print a_donation	
		pointer_loc += 1 

print DEM_state_list, len(DEM_state_list)

## Used for some part of the homework, disabled.
GOP_state_counter = []
GOP_state_variance = []
GOP_state_mean = []
for state_name in GOP_state_list:
	#print state_name
	print 'GOP Checking state: ', state_name
	with open('GOP_donations_2012.csv','rb') as GOP_file:
		GOP_reader = csv.reader(GOP_file)
		pointer_loc = 0
		state_donation_counter = 0
		state_donations = []
		for row in GOP_reader:
			if pointer_loc > 0:
				if float(row[1]) > 0:
					if str(row[2]) == state_name:
						state_donations.append(float(row[1]))
						state_donation_counter += 1
			pointer_loc += 1

	GOP_state_counter.append(state_donation_counter)
	tmp_state_variance = np.var(state_donations)
	tmp_state_mean = np.mean(state_donations)
	GOP_state_variance.append(tmp_state_variance)
	GOP_state_mean.append(tmp_state_mean)

print 'GOP state counter: ', GOP_state_counter
print 'GOP state variance: ', GOP_state_variance
print 'GOP state mean: ', GOP_state_mean
	#GOP_state_variance.append(var(state_donations))		

DEM_state_counter = []
DEM_state_variance = []
DEM_state_mean = []
for state_name in DEM_state_list:
	#print state_name
	print 'DEM Checking state: ', state_name
	with open('DEM_donations_2012.csv','rb') as DEM_file:
		DEM_reader = csv.reader(DEM_file)
		pointer_loc = 0
		state_donation_counter = 0
		state_donations = []
		for row in DEM_reader:
			if pointer_loc > 0:
				if float(row[1]) > 0:
					if str(row[2]) == state_name:
						state_donations.append(float(row[1]))
						state_donation_counter += 1
			pointer_loc += 1

	DEM_state_counter.append(state_donation_counter)
	tmp_state_variance = np.var(state_donations)
	tmp_state_mean = np.mean(state_donations)
	DEM_state_variance.append(tmp_state_variance)
	DEM_state_mean.append(tmp_state_mean)

print 'DEM state counter: ', DEM_state_counter
print 'DEM state variance: ', DEM_state_variance
print 'DEM state mean: ', DEM_state_mean			

state_td_list = []
state_name_list = []
gop_cnt = 0
for gop_state_name in GOP_state_list:
	dem_cnt = 0
	for dem_state_name in DEM_state_list:
		if gop_state_name == dem_state_name:
			n1 = GOP_state_counter[gop_cnt]
			s1 = GOP_state_variance[gop_cnt]
			mu1 = GOP_state_mean[gop_cnt]
			n2 = DEM_state_counter[dem_cnt]
			s2 = DEM_state_variance[dem_cnt]
			mu2 = DEM_state_mean[dem_cnt]
			#print 'The GOP state is', GOP_state_list[gop_cnt]
			#print 'The DEM state is', DEM_state_list[dem_cnt]
			tmp_SE = (s1/n1 + s2/n2)**(0.5)
			td = (mu1 - mu2)/tmp_SE 
			state_name_list.append(gop_state_name)
			state_td_list.append(td)
			print 'The state is: ', GOP_state_list[gop_cnt], ' The td value is: ', td
		dem_cnt += 1
	gop_cnt += 1		


print GOP_state_counter
print GOP_state_list
print DEM_state_counter
print DEM_state_list
#print state_name_list
#print state_td_list
#print len(state_name_list)
#state_name_list.sort()
threshold = 1.65
cnt = 0

DEM_win_state = []
GOP_win_state = []
for state in state_name_list:
	if state_td_list[cnt] > threshold:
		DEM_win_state.append(state)
	else:
		GOP_win_state.append(state)
	cnt += 1	

GOP_win_state.sort()
DEM_win_state.sort()
print 'GOP win argumenet states: ', GOP_win_state
print 'DEM win arguement states: ', DEM_win_state	

# GOP Claim
GOP_threshold = -1.65
cnt = 0
GOP_claim_GOP_win = []
GOP_claim_DEM_win = []
for state in state_name_list:
	if state_td_list[cnt] < GOP_threshold:
		GOP_claim_GOP_win.append(state)
	else:
		GOP_claim_DEM_win.append(state)
	cnt += 1 
GOP_claim_GOP_win.sort()
GOP_claim_DEM_win.sort()	
print 'GOP claim GOP win argumenet states: ', GOP_claim_GOP_win
print 'GOP claim DEM win arguement states: ', GOP_claim_DEM_win


# Create GOP candidate list
print 'now work on candidate'
with open('GOP_donations_2012.csv','rb') as GOP_file:
	GOP_reader = csv.reader(GOP_file)
	pointer_loc = 0
	GOP_candidate_list = []
	for row in GOP_reader:
		if pointer_loc > 0:
			tmp_candidate = int(row[0])
			if tmp_candidate in GOP_candidate_list:
				tmp_candidate
			else:
				GOP_candidate_list.append(tmp_candidate)
			# if pointer_loc < 2:
			# 	tmp_state = str(row[2])
			# 	old_state = tmp_state
			# 	state_list.append(old_state)
			# 	print old_state
			# else:
			# 	tmp_state = str(row[2])
			# 	if tmp_state == old_state:
			# 		tmp_state
			# 	else:
			# 		old_state = tmp_state
			# 		state_list.append(old_state)

			#tmp_state = str(row[2])		
			#print a_donation	
		pointer_loc += 1 

print GOP_candidate_list, len(GOP_candidate_list)

# Create DEM candidate list
with open('DEM_donations_2012.csv','rb') as DEM_file:
	DEM_reader = csv.reader(DEM_file)
	pointer_loc = 0
	DEM_candidate_list = []
	for row in DEM_reader:
		if pointer_loc > 0:
			tmp_candidate = int(row[0])
			if tmp_candidate in DEM_candidate_list:
				tmp_candidate
			else:
				DEM_candidate_list.append(tmp_candidate)
			# if pointer_loc < 2:
			# 	tmp_state = str(row[2])
			# 	old_state = tmp_state
			# 	state_list.append(old_state)
			# 	print old_state
			# else:
			# 	tmp_state = str(row[2])
			# 	if tmp_state == old_state:
			# 		tmp_state
			# 	else:
			# 		old_state = tmp_state
			# 		state_list.append(old_state)

			#tmp_state = str(row[2])		
			#print a_donation	
		pointer_loc += 1 

print DEM_candidate_list, len(DEM_candidate_list)

## Commented out once part (vi)-(i) is done.
# GOP_candidate_sum = []
# for candidate_code in GOP_candidate_list:
# 	#print state_name
# 	print 'GOP Checking candidate: ', candidate_code
# 	with open('GOP_donations_2012.csv','rb') as GOP_file:
# 		GOP_reader = csv.reader(GOP_file)
# 		pointer_loc = 0
# 		candidate_donations = []
# 		for row in GOP_reader:
# 			if pointer_loc > 0:
# 				if float(row[1]) > 0:
# 					if int(row[0]) == candidate_code:
# 						candidate_donations.append(float(row[1]))
# 			pointer_loc += 1
# 	tmp_sum_candidate = np.sum(candidate_donations)		
# 	GOP_candidate_sum.append(tmp_sum_candidate)		

# print len(GOP_candidate_sum), GOP_candidate_sum	

# DEM_candidate_sum = []
# for candidate_code in DEM_candidate_list:
# 	#print state_name
# 	print 'DEM Checking candidate: ', candidate_code
# 	with open('DEM_donations_2012.csv','rb') as DEM_file:
# 		DEM_reader = csv.reader(DEM_file)
# 		pointer_loc = 0
# 		candidate_donations = []
# 		for row in DEM_reader:
# 			if pointer_loc > 0:
# 				if float(row[1]) > 0:
# 					if int(row[0]) == candidate_code:
# 						candidate_donations.append(float(row[1]))
# 			pointer_loc += 1
# 	tmp_sum_candidate = np.sum(candidate_donations)		
# 	DEM_candidate_sum.append(tmp_sum_candidate)		

# print len(DEM_candidate_sum), DEM_candidate_sum	

# s1 = np.var(GOP_candidate_sum)
# n1 = len(GOP_candidate_sum)
# mu1 = np.mean(GOP_candidate_sum)

# s2 = np.var(DEM_candidate_sum)
# n2 = len(DEM_candidate_sum)
# mu2 = np.mean(DEM_candidate_sum)
# SE_candidate = (s1/n1 + s2/n2)**(0.5)
# td_candidate = (mu1 - mu2)/SE_candidate
# print 'The td value considering the candidate is: ', td_candidate
# print 'The GOP mean is: ', mu1
# print 'The DEM mean is: ', mu2

#DEM_state_list = ['NJ']
#GOP_state_list = ['NJ']

## Final part of the homework
candidate_state_td_list = []
candidate_state_name_list = []
gop_cnt = 0
for gop_state_name in GOP_state_list:
	dem_cnt = 0
	for dem_state_name in DEM_state_list:
		# now starting counting for each state
		if gop_state_name == dem_state_name:

			with open('GOP_donations_2012.csv','rb') as GOP_file:
				GOP_reader = csv.reader(GOP_file)
				pointer_loc = 0
				old_candidate = 0
				gop_candidate_donations = []
				tmp_candidate_donations = []
				for row in GOP_reader:
					if pointer_loc > 0:
						if float(row[1]) > 0:
							# if int(row[0]) == candidate_code:
							# 	if str(row[2]) == gop_state_name:
							# 		tmp_candidate_donations.append(float(row[1]))
							# tmp_sum_candidate = np.sum(tmp_candidate_donations)
								tmp_candidate = int(row[0]) 
								if tmp_candidate == old_candidate:										
									tmp_candidate_donations.append(float(row[1]))
								else:
									old_candidate = tmp_candidate
									tmp_sum_candidate = np.sum(tmp_candidate_donations)
									tmp_candidate_donations = []	
									if tmp_sum_candidate > 0:
										gop_candidate_donations.append(tmp_sum_candidate)
									
					pointer_loc += 1					
			# if tmp_sum_candidate > 0:
			# 	gop_candidate_donations.append(tmp_sum_candidate)
			# else:
			# 	print 'Candidate ', candidate_code, 'failed to raise money in ', dem_state_name
										
										
			mu1 = np.mean(gop_candidate_donations)
			s1 = np.var(gop_candidate_donations)
			n1 = len(gop_candidate_donations)							
										
			with open('DEM_donations_2012.csv','rb') as DEM_file:
				DEM_reader = csv.reader(DEM_file)
				pointer_loc = 0
				dem_candidate_donations = []
				tmp_candidate_donations = []
				old_candidate = 0
				for row in DEM_reader:
					if pointer_loc > 0:
						if float(row[1]) > 0:
							if str(row[2]) == dem_state_name:
								tmp_candidate = int(row[0]) 
								if tmp_candidate == old_candidate:										
									tmp_candidate_donations.append(float(row[1]))
								else:
									old_candidate = tmp_candidate
									tmp_sum_candidate = np.sum(tmp_candidate_donations)
									tmp_candidate_donations = []	
									if tmp_sum_candidate > 0:
										dem_candidate_donations.append(tmp_sum_candidate)
									
					pointer_loc += 1
			# if tmp_sum_candidate > 0:
			# 	dem_candidate_donations.append(tmp_sum_candidate)
			# else:
			# 	print 'Candidate ', candidate_code, 'failed to raise money in ', dem_state_name		
			mu2 = np.mean(dem_candidate_donations)
			s2 = np.var(dem_candidate_donations)
			n2 = len(dem_candidate_donations)
			tmp_SE_candidate_state = (s1/n1 + s2/n2)**(0.5)
			tmp_td_candidate_state = (mu1 - mu2)/tmp_SE_candidate_state

			print 'the td score in state: ', dem_state_name, 'is', tmp_td_candidate_state

			candidate_state_name_list.append(gop_state_name)
			candidate_state_td_list.append(tmp_td_candidate_state)

cnt = 0
new_threshold = 1.65
cand_DEM_win_state = []
cand_GOP_win_state = []
for state in candidate_state_name_list:
	if candidate_state_td_list[cnt] > new_threshold:
		cand_DEM_win_state.append(state)
	else:
		cand_GOP_win_state.append(state)
	cnt += 1	

cand_GOP_win_state.sort()
cand_DEM_win_state.sort()
print 'GOP win argumenet states: ', cand_GOP_win_state
print 'DEM win arguement states: ', cand_DEM_win_state	
	

# Using GOP as seed.
# with open('GOP_donations_2012.csv','rb') as GOP_file:
# 	pointer_loc = 0
# 	state_list = []
# 	GOP_reader = csv.reader(GOP_file)
# 	for row in GOP_file:
# 		if pointer_loc > 0:
# 			if pointer_loc < 2:
# 				tmp_state = str(row[2])
# 				old_state = tmp_state
# 				state_list.append(tmp_state)

# 			tmp_state = str(row[2])
# 			print tmp_state
# 			if tmp_state == old_state:
# 				#print 'nothing to be done'
# 				tmp_state
# 			else:
# 				state_list.append(tmp_state)
# 				old_state = tmp_state
# 				#print tmp_state
# 		pointer_loc += 1

# print state_list				





