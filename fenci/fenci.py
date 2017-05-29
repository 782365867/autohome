from sklearn.cross_validation import StratifiedKFold
import numpy as np
x= np.array([[10,11],[20,21],[30,31],[10,11],[20,51],[30,31],[20,21],[30,31],[10,11],[20,21],[30,31],[10,11],[20,51],[30,31],[20,21],[30,31]])
y= np.array([1,1,1,1,1,1,1,2,2,2,2,2])
skf = StratifiedKFold(y,n_folds=3)
for train_index, test_index in skf:
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
print(len(skf))