import csv
import networkx as nx
import numpy as np
# Building the graph

first_column = []
second_column = []
total_nodes = []

with open('edges.csv','rb') as edges:
	edges_reader = csv.reader(edges)
	for row in edges_reader:
		first_column.append(row[0])
		second_column.append(row[1])
		total_nodes.append(row[0])
		total_nodes.append(row[1])

old_node = 1
need_to_check = []
need_to_check.append(old_node)
node_cnt = 0
for a in first_column:
	#print old_node, a
	if int(a) == old_node:
		node_cnt += 1
	else:
		old_node = int(a)
		#print 'change now', node_cnt
		if node_cnt > 10:
			need_to_check.append(old_node)
			
		node_cnt = 0

print len(need_to_check)			


#print len(total_nodes)
node_sets = set(total_nodes)
#print len(node_sets)

G = nx.Graph() 

#final_nodes = []
for a_node in node_sets:
#	final_nodes.append(a_node)
	G.add_node(int(a_node))

#print len(final_nodes)

with open('edges.csv','rb') as edges:
	edges_reader = csv.reader(edges)
	for row in edges_reader:
		G.add_edge(int(row[0]), int(row[1]))	

#print G.number_of_nodes()
#print G.number_of_edges()
#################################################

# AJ_matrix = np.zeros((len(node_sets), len(node_sets)))
# print AJ_matrix.shape
# with open('edges.csv','rb') as edges:
# 	edges_reader = csv.reader(edges)
# 	for row in edges_reader:
# 		index_a = final_nodes.index(row[0])
# 		index_b = final_nodes.index(row[1])
# 		AJ_matrix[index_a, index_b] = 1

# with open("adj_matrix.csv", "wb") as f:
#     writer = csv.writer(f)
#     writer.writerows(AJ_matrix)		

AJ_matrix = nx.adjacency_matrix(G)

#print AJ_matrix(0,0)
print AJ_matrix.shape
# for ii in range(0, 1):
# 	tmp_missing_links = []
# 	for jj in range(0, len(node_sets)):
# 		if AJ_matrix[ii,jj] == 0:
# 			tmp_missing_links.append((int(final_nodes[ii]),int(final_nodes[jj])))
# 			print (int(final_nodes[ii]),int(final_nodes[jj]))
preds = nx.adamic_adar_index(G, ebunch=None)


print max(need_to_check)
#initial_node = 1
missing_links = []
old_u = 1
tmp_missing_links_score = []
tmp_missing_links_u = []
tmp_missing_links_v = []
for u,v,score in preds:

	if int(u) in need_to_check[1000:2000]:
		#print u, old_u 
		if int(u) == old_u:
			#print executed
			tmp_missing_links_score.append(score)
			tmp_missing_links_u.append(u)
			tmp_missing_links_v.append(v)
		else:
			# Need to start sorting now:
			score_sort = np.asarray(tmp_missing_links_score)
			u_sort = np.asarray(tmp_missing_links_u)
			v_sort = np.asarray(tmp_missing_links_v)
			#print len(score_sort), len(u_sort), len(v_sort)
			return_score = score_sort[score_sort.argsort()[-5:]]
			return_u = u_sort[score_sort.argsort()[-5:]]
			return_v = v_sort[score_sort.argsort()[-5:]]
			for ii in range(0,5):
				missing_links.append([return_u[ii],return_v[ii]])
				print return_u[ii], return_v[ii], return_score[ii], 'length missing links', len(missing_links)
			old_u = int(u)
			del tmp_missing_links_score, tmp_missing_links_u, tmp_missing_links_v
			tmp_missing_links_score = []
			tmp_missing_links_u = []
			tmp_missing_links_v = []
			if int(u) == old_u:
				tmp_missing_links_score.append(score)
				tmp_missing_links_u.append(u)
				tmp_missing_links_v.append(v)  		
  		

with open('missing_links_submission_c2.csv', 'ab') as f:                                    
    writer = csv.writer(f)                                                       
    writer.writerows(missing_links)	
	


# G=nx.path_graph(5)
# print G
# print(nx.shortest_path(G,source=0,target=4))