# Load the dataset
import numpy as np
import urllib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets, cross_validation
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score, accuracy_score

#Loading the main datasets...
import h5py
dataset = h5py.File('../ml_proj1_data.h5', 'r')
print(dataset.keys())


#Extract the general x values
trainx = dataset['xtrain']
vx = dataset['xval']
testx = dataset['xtest']

#Extract the data for this part
trainy = dataset['bin_train']
vy = dataset['bin_val']
print("%s\n%s\n%s\n%s\n%s" %(trainx, vx, testx, trainy, vy))


###############################################################################

#transform = SelectFromModel(LinearSVC(penalty="l1", dual=False, ))
clf = svm.SVC(kernel='linear')

#transform the data
lsvc = LinearSVC(C=1, penalty="l1", dual=False).fit(trainx, trainy)
model = SelectFromModel(lsvc, prefit=True)
trainx_new = model.transform(trainx)
vx_new = model.transform(vx)
testx_new = model.transform(testx)
print("After Feature Selection: %s_%s_%s"
      %(trainx_new.shape, vx_new.shape, testx_new.shape))

#clf_fs = Pipeline([  ('fs', transform),('clf', clf)])


###############################################################################
# Set the parameters by cross-validation

validation_means = list()
validation_stds = list()


validation_accmeans = list()
validation_accstds = list()

reg_param = (0.001, 0.01, 1, 10, 100)

for reg in reg_param:
    print(reg)
    clf.set_params(C=reg)
    clf.fit(trainx_new, trainy)
    # Compute cross-validation score using all CPUs
    v_predict = clf.predict(vx_new);
    f_scores = f1_score(v_predict, vy)
    a_scores = accuracy_score(v_predict, vy)
    print (f_scores.mean())    
    validation_means.append(f_scores.mean())
    validation_stds.append(f_scores.std())
    validation_accmeans.append(a_scores.mean())
    validation_accmeans.append(a_scores.std())
 
print("Linear")
print(reg_param)
print(validation_means)
print(validation_stds)
print(validation_accmeans)
print(validation_accmeans)
#plt.errorbar(reg_param, validation_means, np.array(validation_stds))
#plt.title('Performance of the Linear SVM varying the regularization parameter')
#plt.xlabel('Reg_param')
#plt.ylabel('Prediction rate')
#plt.axis('tight')
#plt.show()

#Testing the 
clf.set_params(kernel='rbf', gamma='auto')
validation_means = list()
validation_stds = list()

validation_accmeans = list()
validation_accstds = list()


for reg in reg_param:
    print(reg)
    clf.set_params(C=reg)
    clf.fit(trainx_new, trainy)
    # Compute cross-validation score using all CPUs
    v_predict = clf.predict(vx_new);
    f_scores = f1_score(v_predict, vy)
    a_scores = accuracy_score(v_predict, vy)
    print (f_scores.mean())    
    validation_means.append(f_scores.mean())
    validation_stds.append(f_scores.std())
    validation_accmeans.append(a_scores.mean())
    validation_accmeans.append(a_scores.std())
 
print("RBF")
print(reg_param)
print(validation_means)
print(validation_stds)
print(validation_accmeans)
print(validation_accmeans)

#Best settings:
clf.set_params(kernel='rbf', gamma='auto', C=100)
clf.fit(trainx_new, trainy)
t_predict = clf.predict(testx_new);

f = h5py.File('bin_test.h5','a')
dset = f.create_dataset('svm',data=t_predict)




