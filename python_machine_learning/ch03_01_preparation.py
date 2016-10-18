
from sklearn import datasets
import numpy as np

# load iris data
iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target

print "="*50
print "Iris dataset shape:"
print "  (X {} ) (y {} \n".format(X.shape, y.shape)

# split train/test data
from sklearn.model_selection import  train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=0)
print "="*50
print "Split Result:"
print "    X_train : {} y_train : {}".format(X_train.shape, y_train.shape)
print "    X_test : {} y_test : {}\n".format(X_test.shape, y_test.shape)

# standard scaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
print "="*50
print "Standard Scaling:"
print "    Pre-Scaling : \n", X_train[:3]
print "    Post-Scaling : \n{}\n".format(X_train_std[:3])
print "    SC.mean\n", sc.mean_
print "    SC.variance\n", sc.scale_
