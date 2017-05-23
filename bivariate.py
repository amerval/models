%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import plotly.plotly as py
import seaborn as sns
import math
from scipy.stats import norm, expon

#######################################################################
# first generate a sample: one bi-variate distribution for one class,
# one constructed one for the other class
nbs = 750
mu = [1.7,2.8]
sig = np.array([5,4])
tt = 0.5
amin = -1
amax = 4.5
noise = 0.6

#X = np.random.normal(loc=mu, scale=sig)
cov = [[1,tt],[tt,0.8]]
x,y = np.random.multivariate_normal(mean=mu, cov=cov, size=nbs).T

x2 = np.random.uniform(amin, amax, size=nbs)
y2 = 0.7 + x2**2/6 + np.random.normal(loc=0, scale=noise, size=nbs)

plt.plot(x2,y2, 'o', color='blue', alpha=0.4)
plt.plot(x,y,'o',color='red', alpha=0.4)

plt.axis('equal')
plt.show()

#######################################################################
# construct data set
aa = np.array([x,y,np.ones(nbs,)]).T
bb = np.array([x2,y2,np.zeros((nbs,))]).T
X = np.concatenate((aa,bb))

mydf = pd.DataFrame(data=X, columns=['x1','x2','class'])
mydf.head()

#######################################################################
# split train/valid   
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
         mydf.iloc[:,0:2], mydf.iloc[:,2], test_size=0.3, random_state=0)
         
# as well, we'll create a standardised version of the input set 
#    for comparison of performances
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
# apply the transformation
X_std_train = sc.transform(X_train)
X_std_test = sc.transform(X_test)
      
#######################################################################         
# logistic regression
from sklearn.linear_model import LogisticRegression as logreg

mylogreg = logreg(solver='lbfgs') # all other params to default
mylogreg.fit(X_train, y_train)
pred = mylogreg.predict(X_test)
# note: as with Perceptron, the predictions are 0 or 1
error = (y_test != pred)
print('Misclass: ', error.sum())
print('Misclass rate: ', format(error.sum()/error.shape[0]*100, '4.2f'), '%')

# note that you can access the estimated probabilities
predprob = mylogreg.predict_proba(X_test)
print(predprob[1:10,:])   

intercept = -1*mylogreg.intercept_[0]/mylogreg.coef_[0][1]
slope = -1*mylogreg.coef_[0][0]/mylogreg.coef_[0][1]
print("with log reg: x2 = ",intercept," + ", slope,"x1")

#######################################################################   
# neural network
from sklearn.neural_network import MLPClassifier as ann_
from sklearn.neural_network import MLPRegressor as ann

# 1. simple perceptron: only one neuron + output layer
# this 'network' has two neurons: one hidden neuron (logistic) + one output layer (logistic as well)
simpleNN = ann_(hidden_layer_sizes=[1], activation='logistic', solver='lbfgs', 
               max_iter=100, random_state=0)
simpleNN.fit(X_train, y_train)
pred = simpleNN.predict(X_test)

print('iter to convergence: %d' %simpleNN.n_iter_)

error = (y_test != pred)
print('Simple Perceptron:')
print('Misclass: ', error.sum())
print('Misclass rate: ', format(error.sum()/error.shape[0]*100, '4.2f'), '%')

# now, this one has one hidden neuron (layer) + one linear output
#    drawback: the cost function is loess
simpleNN2 = ann(hidden_layer_sizes=[1], activation='logistic', solver='lbfgs', 
               max_iter=100, random_state=0)
simpleNN2.fit(X_train, y_train)
out = simpleNN2.predict(X_test)
# note: we have to convert the predictions to binaries manually
v_error = [0]*10
print('One sigmoid neuron + linear output:')
for thr in range(1,11):
    tt = (out > 1-thr/10)
    print('   with threshold %3.1f' %(1-thr/10):')
    errortmp = (y_test != tt)
    print('   Misclass rate: ', format(errortmp.sum()/errortmp.shape[0]*100, '4.2f'), '%')
    
# 2. proper network: find optimal NHU
ntry = 5  # number of tries with different init
v_train = [0]*nmax
v_error = [0]*nmax
emin_tot = 1500
for ii in range(1,nmax+1):
    emin = 1500
    #multiple init
    for _ in range(1,ntry+1):
        NN = ann_(hidden_layer_sizes=[ii], activation='logistic', solver='lbfgs', 
                   max_iter=100)
        NN.fit(X_train, y_train)
        pred = NN.predict(X_train)
        etrain = (y_train != pred)
        # keep if error smaller than previous
        if (etrain.sum() < emin):
            emin = etrain.sum()
            v_train[ii-1] = (etrain.sum())/etrain.shape[0]*100
            pred = NN.predict(X_test)
            error = (y_test != pred)
            v_error[ii-1] = (error.sum())/error.shape[0]*100
            if (error.sum()<emin_tot):
                emin_tot = error.sum()
                bestNN = NN
                bestNHU = ii
                
plt.plot(range(1,len(v_train)+1), v_train,marker='o')
plt.plot(range(1,len(v_error)+1), v_error,marker='o',c='red')
plt.xlabel('NHU')
plt.ylabel('misclass rate')
plt.show()             

pred = bestNN.predict(X_test)
error = (y_test != pred)
print('Neural Network with %d hidden units', %bestNHU)
print('   Misclass: ', error.sum())
print('   Misclass rate: ', format(error.sum()/error.shape[0]*100, '4.2f'), '%')

#######################################################################   
# Support Vector Machines
from sklearn.svm import SVC #Support Vector Classifier

# first: linear model - for comparison
linsvm = SVC(kernel='linear')  # all defaults are good here
linsvm.fit(X_train, y_train)
pred = linsvm.predict(X_test)
# note: as with Perceptron, the predictions are 0 or 1
error = (y_test != pred)
print('linear SVM')
print('   Misclass: ', error.sum())
print('   Misclass rate: ', format(error.sum()/error.shape[0]*100, '4.2f'), '%')

# interesting feature of SVC: selection of input samples of interest 
#    the so-called 'support vectors'
print('   nb SV: ', linsvm.support_.shape)

# now proper non linear SVM - RBF kernel
mysvm = SVC()  # all defaults are good here
mysvm.fit(X_train, y_train)
pred = mysvm.predict(X_test)
# note: as with Perceptron, the predictions are 0 or 1
error = (y_test != pred)
print('non-linear SVM - RBF kernel')
print('   Misclass: ', error.sum())
print('   Misclass rate: ', format(error.sum()/error.shape[0]*100, '4.2f'), '%')

# support vectors
print('   nb SV: ', mysvm.support_.shape)

#######################################################################   
# Decision trees
from sklearn.tree import DecisionTreeClassifier as dectree

# first: let's just try one node for comparison with above
myleaf = dectree(criterion='entropy', max_depth=1, random_state=0)

myleaf.fit(X_train,y_train)
pred = myleaf.predict(X_test)

error = (y_test != pred)
print('Decision Tree - single leaf')
print('   Misclass: ', error.sum())
print('   Misclass rate: ', format(error.sum()/error.shape[0]*100, '4.2f'), '%')

# let's see how the rate of misclass evolves with the depth
nmax = 12
v_error = [0]*nmax
emin = 1500
for ii in range(1,nmax+1):
    mytree = dectree(criterion='entropy', max_depth=ii, random_state=0)
    mytree.fit(X_train,y_train)
    pred = mytree.predict(X_test)
    error = (y_test != pred)
    v_error[ii-1] = (error.sum())/error.shape[0]*100
    if (error.sum()<emin):
        emin = error.sum()
        bestdepth = ii
    
plt.plot(range(1,len(v_error)+1), v_error,marker='o')
plt.xlabel('tree depth')
plt.ylabel('misclass rate')
plt.show()

mytree = dectree(criterion='entropy', max_depth=bestdepth, random_state=0)
mytree.fit(X_train,y_train)
pred = mytree.predict(X_test)
error = (y_test != pred)
print('Decision Tree - depth = %d' %bestdepth)
print('   Misclass: ', error.sum())
print('   Misclass rate: ', format(error.sum()/error.shape[0]*100, '4.2f'), '%')
