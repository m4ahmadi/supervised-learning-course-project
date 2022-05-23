# Load the dataset
import numpy as np
import urllib
import numpy as np
from sklearn import svm, datasets, cross_validation
from sklearn.svm import LinearSVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
#Loading the main datasets...
import h5py
dataset = h5py.File('../ml_proj1_data.h5', 'r')
print(dataset.keys())
#Extract the general x values
trainx = dataset['xtrain']
vx = dataset['xval']
testx = dataset['xtest']

#Extract the data for this part
#trainy = dataset['reg_train']
vy = dataset['reg_val']

#separate data file
f = open("reg_train.txt")
trainy = f.readlines()
trainy = np.array(trainy, dtype = 'f8')
print("%s\n%s\n%s\n%s" %(trainx, vx, testx, vy))


###############################################################################
#transform = SelectFromModel(LinearSVC(penalty="l1", dual=False, ))
clf = Ridge(alpha = 0.1)

#transform the data
lsvc = linear_model.Lasso(alpha = 0.001).fit(trainx, trainy)
model = SelectFromModel(lsvc, prefit=True)
trainx_new = model.transform(trainx)
vx_new = model.transform(vx)
testx_new = model.transform(testx)
print("After Feature Selection: %s_%s_%s"
      %(trainx_new.shape, vx_new.shape, testx_new.shape))

#clf_fs = Pipeline([  ('fs', transform),('clf', clf)])


###############################################################################
# Set the parameters by cross-validation

validation_smeans = list()
validation_ameans = list()

r_param = (0.001, 0.01, 0.1, 1, 10, 100, 1000)

for r in r_param:
    print(r)
    clf.set_params(alpha=r)
    clf.fit(trainx_new, trainy)
    v_predict = clf.predict(vx_new);
    s_scores = math.sqrt(mean_squared_error(v_predict, vy))
    a_scores = mean_absolute_error(v_predict, vy)
    print (a_scores.mean())    
    validation_smeans.append(s_scores)
    validation_ameans.append(a_scores)
 
print("Linear")
print(r_param)
print(validation_smeans)
print(validation_ameans)


#Testing the 
clf = KernelRidge(kernel='rbf')


validation_smeans = list()
validation_ameans = list()


for r in r_param:
    print(r)
    clf.set_params(alpha=r)
    clf.fit(trainx_new, trainy)
    v_predict = clf.predict(vx_new);
    s_scores = math.sqrt(mean_squared_error(v_predict, vy))
    a_scores = mean_absolute_error(v_predict, vy)
    print (a_scores.mean())    
    validation_smeans.append(s_scores)
    validation_ameans.append(a_scores)
 
print("Kernelized")
print(r_param)
print(validation_smeans)
print(validation_ameans)




#Best settings:
clf = KernelRidge(kernel='rbf', alpha=0.001)
clf.fit(trainx_new, trainy)
t_predict = clf.predict(testx_new);

f = h5py.File('reg_test.h5','a')
dset = f.create_dataset('linear', data=t_predict)




