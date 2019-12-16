from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import plot_model
from keras import callbacks
import numpy as np
import tensorflow as tf
from generalised_data_generation import *
import datetime
import keras.backend as K
import pickle
import networkx as nx
from collections import deque

no_of_nodes = 15
data_length = 100
# Noise parameters
mu    = 0
sigma = 10

# coefficients range
coeff_lo = -10
coeff_hi = 10

# Number of categories
cat_num = 10

# Number of epochs used for training model
model_train_epochs = 15

# train for 1st variable
def define_model():
    model = Sequential()
    model.add(Dense(12, input_shape=(no_of_nodes - 1, cat_num), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Flatten())
    model.add(Dense(cat_num, activation='sigmoid'))
    return model

def construct_data(indx, x_1_hot):
    x_data = np.zeros((data_length, no_of_nodes - 1, cat_num))
    itr = 0
    for i in x_1_hot.keys():        
        if i != indx:
            for j in range(data_length):
                x_data[j, itr, :] = x_1_hot[i][j, :]
            
            itr += 1
    return x_data
        

class PauseLearningKeras(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        x0 = model.layers[0].get_weights()
        y0 = model.layers[0].output
        x1 = model.layers[1].get_weights()
        y1 = model.layers[1].output
        
        #input('Yes it reallu stopped 8)')
    
def train_model_for_index(idx, x_1_hot):
    x_temp = construct_data(idx, x_1_hot)
    y_temp = x_1_hot[idx]
    x_data = x_temp
    y_data = y_temp
    
    K.clear_session()
    model = define_model()
    
    '''
    # Define Callbacks
    logdir ="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, write_graph=True, write_grads=True)
    pasue_on_callback = PauseLearningKeras()
    '''
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.fit(x_data, y_data, epochs=10, validation_split = 0.1, callbacks=[tensorboard_callback, pasue_on_callback]) use this if callback is required
    model.fit(x_data, y_data, epochs=model_train_epochs, validation_split = 0.1)
    
    
    # Not saving weights for the moment
    #model.save_weights("model_for_idx_24.h5")
    model_json = model.to_json()
    model_name = 'model_' + str(idx) + 'json'
    with open(model_name, "w") as json_file:
        json_file.write(model_json)
    
    return model
    
def calculate_loss_for_variables(trained_models, x_1_hot):
    scores1 = {}
    for idx in range(no_of_nodes):
        x_temp = construct_data(idx, x_1_hot)
        y_temp = x_1_hot[idx]
        x_data = x_temp
        y_data = y_temp
        scores1[idx] = trained_models[idx]['model'].evaluate(x_data, y_data, verbose=1)
        #print('For node ', idx, 'score is ', scores1, ' And metrics is ', trained_models[idx]['model'].metrics_names )
    
    # Now seprately print results
    for k in scores1.keys():
        print('For node ', k, 'score is ', scores1[k], ' And metrics is ', trained_models[k]['model'].metrics_names )
    print('done')
    
    
    
    
################# Now detect intrvention ##############################
def update_data_for_softintervention():
    with open('data_analyse.pkl', 'rb') as f:
        x_1_hot, x_final_dict, G = pickle.load(f)
    
    idx = 5
    ###### Do soft intervene on idx node
    # Get incoming edges
    x = G.in_edges(idx)
    G.node[idx]['data'] = np.random.normal(mu, sigma, data_length)
    for i in x:
        if i[1] != idx:
            input('Bro You screwed up......  :|')
        G[i[0]][i[1]]['weight'] = G[i[0]][i[1]]['weight'] * np.random.uniform(-2, 2)
        G.node[i[1]]['data'] += G.node[i[0]]['data'] * G[i[0]][i[1]]['weight']
    
    # Initialize x_dict, used for updating graph
    keys = list(G.nodes)
    x_dict = dict(zip(keys, [None]*len(keys)))
    x_dict = copy_graph_to_x_dict(x_dict, G)
    
    q_list = deque()
    G, q_list = update_all_subsequent_nodes(idx, G, q_list, x_dict)
    x_dict = copy_graph_to_x_dict(x_dict, G)
    # verifying that after this step as well all graph nodes have correct values is remaining
    return G, x_dict
    
    # select 1 random node
def main():
    
    G = nx.Graph()
    x_1_hot, x_final_dict, G, bin_limits = return_required_results(no_of_nodes, data_length, mu, sigma, coeff_lo, coeff_hi, cat_num)
    with open('data_analyse.pkl', 'wb') as f:
        pickle.dump([x_1_hot, x_final_dict, G], f)
    
    trained_models = {}
    # Train model for each variable
    for i in range(no_of_nodes):
        trained_models[i] = {"model":train_model_for_index(i, x_1_hot)}
    
    G, x_dict = update_data_for_softintervention()
    x_1_hot, x_final_dict = construct_1_hot_vector(x_dict, bin_limits, G, data_length, cat_num)
    calculate_loss_for_variables(trained_models, x_1_hot)
    
if __name__ == "__main__": 
    main()
    
    

