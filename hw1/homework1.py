import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import copy
####QUESTION 1####
## a
data = pd.read_csv('real_estate.csv')
raw_data = data.copy()
lenth = data.shape[0]
index_in_file = []
index_in_code = []
for i in range(lenth):
    if data.iloc[i].isnull().any():
        index_in_file.append(i + 2)
        index_in_code.append(i)
print('The index of removed rows in the file are ',index_in_file)
data = data.drop(index = data.index[index_in_code])
data = data.reset_index(drop = False)
data = data.drop(['transactiondate', 'latitude', 'longitude', 'price'], axis = 1)

##b
features = ['age', 'nearestMRT', 'nConvenience']
for feature in features:
    minx = min(data[feature])
    maxx = max(data[feature])
    for i in data.index:
        data.loc[i, feature] = (data.loc[i,feature] - minx) / (maxx - minx)
mean_feature = {}
for feature in features:
    mean_feature[feature] = np.mean(data[feature])
    print('The mean value of feature \'', feature, '\' = ', mean_feature[feature])

####QUESTION 2####
lenth = data.shape[0]
cut = lenth // 2
lenth = data.shape[0]
cut = lenth // 2

print('The first row of the training set is \n', data.loc[0][1:],
      '\nThe last row of the training set is \n', data.loc[cut - 1][1:],
      '\nThe first row of the testing set is \n', data.loc[cut][1:],
      '\nThe last row of the tesing set is \n', data.loc[cut+cut-1][1:])

#split train set and test set
X = np.zeros((lenth,1,4))
for i in data.index:
    X[i] = [1,data.loc[i, 'age'], data.loc[i,'nearestMRT'], data.loc[i, 'nConvenience']]

y = np.zeros((lenth,1))
for i in range(len(X)):
    raw_index = data.loc[i, 'index']  #the index in raw_data
    y[i] = raw_data.loc[raw_index, 'price']

X_train = X[0 : cut]
X_test = X[cut : cut + cut]
y_train = y[0 : cut]
y_test = y[cut : cut + cut]


####QUESTION 5####
##a
losses = []

fig, ax = plt.subplots(3, 3, figsize=(10, 10))
nIter = 400
alphas = [10, 5, 2, 1, 0.5, 0.25, 0.1, 0.05, 0.01]

for eta in alphas:
    w0 = np.ones((4, 1))
    loss_per_iter = []
    for iteration in range(nIter):
        loss_all_row = 0
        partial_matrix = np.zeros((4, 1))
        for i in range(len(X_train)):
            dot_item = float(np.dot(X_train[i], w0))
            loss_all_row += (1 / 4 * (float(y[i]) - dot_item) ** 2 + 1) ** 0.5 - 1
            # update w0
            for j in range(len(partial_matrix)):  # for each w in w0
                partial_matrix[j] += X_train[i][0][j] * (dot_item - float(y[i])) / (
                            2 * (((dot_item - float(y[i])) ** 2 + 4)) ** 0.5)
        loss_per_iter.append(loss_all_row / len(X_train))
        w0 = w0 - eta * (1 / len(X_train)) * partial_matrix
    losses.append(loss_per_iter)

for i, ax in enumerate(ax.flat):
    ax.plot(losses[i])
    ax.set_title(f"step size: {alphas[i]}")
plt.tight_layout()
plt.show()

# for i in range(len(losses)):
#     print('step size = ', alphas[i])
#     print(losses[i][-10:])

##c
eta = 0.3
w0 = np.ones((4,1))
w_list = []
w_list.append(w0)
for iteration in range(nIter):
    loss_all_row = 0
    partial_matrix = np.zeros((4,1))
    for i in range(len(X_train)):
        dot_item = float(np.dot(X_train[i], w0))
        loss_all_row += (1/4 * (float(y_train[i]) - dot_item)**2 + 1)**0.5 - 1
        #update w0
        for j in range(len(partial_matrix)):   #for each w in w0
            partial_matrix[j] += X_train[i][0][j]*(dot_item - float(y_train[i])) / (2*(((dot_item - float(y_train[i]))**2 + 4))**0.5)
    w0 = w0 - eta * (1/len(X_train)) * partial_matrix
    w_list.append(w0)

print('The final weight vector is: \n', w_list[-1])
w_list = np.array(w_list)
w_t = w_list.T[0]
for i in range(len(w_t)):
    plt.plot(w_t[i], label = ['w_0', 'w_1', 'w_2', 'w_3'][i])
    plt.legend()
plt.show()

train_loss = 0
w0 = w_list[-1]
for i in range(len(X_train)):
    dot_item = float(np.dot(X_train[i], w0))
    train_loss += (1/4 * (float(y_train[i]) - dot_item)**2 + 1)**0.5 - 1
print('The loss on the train set is ', train_loss / len(X_train))

test_loss = 0
for i in range(len(X_test)):
    dot_item = float(np.dot(X_test[i], w0))
    test_loss += (1/4 * (float(y_test[i]) - dot_item)**2 + 1)**0.5 - 1
print('The loss on the test set is ', test_loss/len(X_test))

####QUESTION 6####
##a
epoch = 6
losses = []

fig, ax = plt.subplots(3, 3, figsize=(10, 10))
nIter = 400
alphas = [10, 5, 2, 1, 0.5, 0.25, 0.1, 0.05, 0.01]

for eta in alphas:
    w0 = np.ones((4, 1))
    loss_per_eta = []
    partial_matrix = np.zeros((4, 1))
    for iteration in range(epoch):
        for i in range(len(X_train)):
            dot_item = float(np.dot(X_train[i], w0))
            # update w0
            for j in range(len(partial_matrix)):  # for each w in w0
                partial_matrix[j] = X_train[i][0][j] * (dot_item - float(y_train[i])) / (
                            2 * (((dot_item - float(y_train[i])) ** 2 + 4)) ** 0.5)
            w0 = w0 - eta * partial_matrix

            loss_all_row = 0
            for k in range(len(X_train)):
                dot_item = float(np.dot(X_train[k], w0))
                loss_all_row += (1 / 4 * (float(y_train[k]) - dot_item) ** 2 + 1) ** 0.5 - 1
            loss_per_eta.append(loss_all_row / len(X_train))
    losses.append(loss_per_eta)

for i, ax in enumerate(ax.flat):
    ax.plot(losses[i])
    ax.set_title(f"step size: {alphas[i]}")  # plot titles
plt.tight_layout()  # plot formatting
plt.show()

##c
w0 = np.ones((4,1))
eta = 0.4
w_list = []
w_list.append(w0)
loss_per_eta = []
partial_matrix = np.zeros((4,1))
for iteration in range(epoch):
    for i in range(len(X_train)):
        dot_item = float(np.dot(X_train[i], w0))
        #update w0
        for j in range(len(partial_matrix)):   #for each w in w0
            partial_matrix[j] = X_train[i][0][j]*(dot_item - float(y_train[i])) / (2*(((dot_item - float(y_train[i]))**2 + 4))**0.5)
        w0 = w0 - eta * partial_matrix
        w_list.append(w0)
w_list = np.array(w_list)
w_t = w_list.T[0]
for i in range(len(w_t)):
    plt.plot(w_t[i], label = ['w_0', 'w_1', 'w_2', 'w_3'][i])
    plt.legend()
plt.show()

train_loss = 0
w0 = w_list[-1]
print('The final model is \n', w0)
for i in range(len(X_train)):
    dot_item = float(np.dot(X_train[i], w0))
    train_loss += (1/4 * (float(y_train[i]) - dot_item)**2 + 1)**0.5 - 1
print('The loss on the train set is ', train_loss / len(X_train))

test_loss = 0
for i in range(len(X_test)):
    dot_item = float(np.dot(X_test[i], w0))
    test_loss += (1/4 * (float(y_test[i]) - dot_item)**2 + 1)**0.5 - 1
print('The loss on the test set is ', test_loss/len(X_test))

