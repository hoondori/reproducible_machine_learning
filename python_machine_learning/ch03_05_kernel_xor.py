from sklearn import datasets
import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler
from ch03_libs import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Generate xor
np.random.seed(0)
X_xor = np.random.randn(200,2)
y_xor = np.logical_xor(X_xor[:,0]>0, X_xor[:,1]>0)
y_xor = np.where(y_xor,1, -1)

plt.scatter(X_xor[y_xor==1,0], X_xor[y_xor==1,1],
            c='b', marker='x', label='1')
plt.scatter(X_xor[y_xor==-1,0], X_xor[y_xor==-1,1],
            c='r', marker='s', label='-1')
plt.title("XOR Problem")
plt.ylim(-3.0)
plt.legend()
plt.show()

# solve by RBF kernel
svm = SVC(kernel='rbf', random_state=0, gamma=01, C=10)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.title("XOR solved by SVM w RBF kernel")
plt.legend(loc='upper left')
plt.show()


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

svm = SVC(kernel='rbf', C=1.0, gamma=0.2, random_state=0)
svm.fit(X_train_std, y_train)
y_pred = svm.predict(X_test_std)
mis_samples = (y_test != y_pred).sum()
print "="*50
print "Misclassified samples: {}".format(mis_samples)
print "Accuracy : 1 - {}/{} = {}\n".format(mis_samples, X_test_std.shape[0], 1 - 1.*mis_samples/X_test_std.shape[0])

# Decision boundary plot
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined_std, y_combined,
                      classifier=svm, test_idx=range(105,150))
plt.title('Decision boundary with Support Vector Machine RBF kernel')
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()