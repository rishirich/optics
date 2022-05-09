# %%
# Imports
import numpy as np
import pandas as pd
import heapq as heap
import matplotlib.pyplot as plt
import matplotlib
import streamlit as st
import time

# %%
# Declarations
UNDEFINED = np.inf

# Class definitions
class Vector:
    def __init__(self,df,i):
        self.index = i
        self.is_processed = False
        self.core_dist = UNDEFINED
        self.reach_dist = UNDEFINED
        self.data = df.loc[i]

    def get_distance(self,neighbour,columns):
        return np.linalg.norm(self.data[columns] - neighbour.data[columns])
        
    def set_core_distance(self,minPts,N,columns):
        # if len(N) < minPts:
            # self.core_dist = self.core_dist
        if minPts <= len(N):
            neighbour_dists = []
            for neighbour in N:
                neighbour_dists.append(self.get_distance(neighbour,columns))
            neighbour_dists.sort()
            self.core_dist = neighbour_dists[minPts-1]
    
    def get_core_distance(self):
        return self.core_dist
    
    def set_reachability_dist(self,reach_dist):
        self.reach_dist = reach_dist
    
    def get_reachability_dist(self):
        return self.reach_dist
    
    def mark_processed(self):
        self.is_processed = True
    
    def is_vector_processed(self):
        return self.is_processed
    
    def __lt__(self, vector2):
        return self.reach_dist < vector2.reach_dist
    
    def __str__(self):
        return 'Vector:' + '\nIndex: ' + str(self.index) + '\nis_processed: ' + str(self.is_processed) + '\nCore Distance: ' + str(self.core_dist) + '\nReachability Distance: ' + str(self.reach_dist) + '\nData: ' + str(self.data) 

# %%
# Functions
def get_neighbours(vectors,current_vector,eps,columns):
    neighbours = []
    for vector in vectors:
        if vector.get_distance(current_vector,columns) <= eps:
            neighbours.append(vector)
    return neighbours

def update_cluster(current_vector,N,seed_vectors,columns):
    core_dist = current_vector.get_core_distance()
    for neighbour in N:
        if not neighbour.is_vector_processed():
            new_reachability_dist = max(core_dist,current_vector.get_distance(neighbour,columns))
            if neighbour.get_reachability_dist() == UNDEFINED:
                neighbour.set_reachability_dist(new_reachability_dist)
                heap.heappush(seed_vectors,neighbour)
            elif neighbour.get_reachability_dist() > new_reachability_dist:
                    neighbour.set_reachability_dist(new_reachability_dist)
                    heap.heapify(seed_vectors)

def optics(vectors,eps,minPts,columns):
    ordered_vectors = list()
    for vector in vectors:
        if not vector.is_vector_processed():
            N = get_neighbours(vectors,vector,eps,columns)
            vector.mark_processed()
            vector.set_core_distance(minPts,N,columns)
            ordered_vectors.append(vector)
            if vector.get_core_distance() != UNDEFINED:
                seed_vectors = []
                update_cluster(vector,N,seed_vectors,columns)
                while len(seed_vectors) > 0:
                    seed_vector = heap.heappop(seed_vectors)
                    N_seed = get_neighbours(vectors,seed_vector,eps,columns)
                    seed_vector.mark_processed()
                    seed_vector.set_core_distance(minPts,N_seed,columns)
                    ordered_vectors.append(seed_vector)
                    if seed_vector.get_core_distance() != UNDEFINED:
                        update_cluster(seed_vector,N_seed,seed_vectors,columns)
    return ordered_vectors

def assign_cluster_numbers(df,ordered_vectors,target_eps):
    cluster_number = 0
    for vector in ordered_vectors:
        if vector.get_reachability_dist() > target_eps:
            if vector.get_core_distance() <= target_eps:
                cluster_number+=1
                df.at[vector.index,'cluster_number'] = cluster_number
        else:
            df.at[vector.index,'cluster_number'] = cluster_number
    return df

def plot_cluster_ordering(df,ordered_vectors):
    cluster_amount = df.cluster_number.max()
    cmap = plt.cm.get_cmap('hsv', cluster_amount)
    points_color_list = []
    X = []
    Y = []
    Z = []
    for vector in ordered_vectors:
        data = df.loc[vector.index]
        X.append([data[2]])
        Y.append([data[3]])
        Z.append([data[4]])
        points_color_list.append(str(matplotlib.colors.rgb2hex(cmap(data.cluster_number))) if data.cluster_number != -1 else '#000000')

    height_data = [vector.get_reachability_dist() if not vector.get_reachability_dist() == UNDEFINED else eps for vector in ordered_vectors]
    X_data = [i + 1 for i in range(len(ordered_vectors))]
    points_path_list = []

    fig = plt.figure(figsize=(10, 15))
    ax = fig.add_subplot(2, 1, 1, projection='3d')
    ax.set(xlabel='Age', ylabel='Annual Income (k$)', zlabel='Spending Score (1-100)')
    for i in range(len(ordered_vectors)):  
        points_path_list.append(ax.plot(X[i],Y[i],Z[i],linestyle='None',marker='o',color='#000000',markersize=6))

    ax2 = fig.add_subplot(2, 1, 2)

    ax2.bar(x=[], height=[], width=1)
    ax2.set_xlim([0, 200])
    ax2.set(xlabel='Customers', ylabel='Reachability distance')
    the_plot = st.pyplot(fig)

    def animate(i):
        points_path_list[i][0].set_color(points_color_list[i])
        ax2.bar(x=X_data[i], height=height_data[i], color=points_color_list[i], width=1)
        the_plot.pyplot(fig)

    for i in range(len(points_color_list)):
        animate(i)

# %%
eps = 9
target_eps = 9
minPts = 3

df = pd.read_csv("Mall_Customers.csv")
df['cluster_number'] = -1
df['color'] = '#000000'
vectors = list()
numerical_columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
for i in range(len(df)):
    vector = Vector(df,i)
    vectors.append(vector)

ordered_vectors = optics(vectors,eps,minPts,numerical_columns)
df=assign_cluster_numbers(df,ordered_vectors,target_eps)
plot_cluster_ordering(df,ordered_vectors)
