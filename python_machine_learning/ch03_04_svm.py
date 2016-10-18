
from sklearn import datasets
import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler
from ch03_libs import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# load iris data
iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target

# split train/test data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=0)

# standard scaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# svm
svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_std, y_train)

y_pred = svm.predict(X_test_std)
mis_samples = (y_test != y_pred).sum()
print "="*50
print "Misclassified samples: {}".format(mis_samples)
print "Accuracy : 1 - {}/{} = {}\n".format(mis_samples, X_test_std.shape[0], 1 - 1.*mis_samples/X_test_std.shape[0])

# Decision boundary plot
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std,
                      y=y_combined,
                      classifier=svm,
                      test_idx=range(105,150))
plt.title('Decision boundary with SVM')
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()