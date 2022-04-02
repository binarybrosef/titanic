##--Random forest model predicting survival outcomes for the Kaggle Titanic challenge--##

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from preprocessing import x_train, y_train, x_test


classifier = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1, max_features='sqrt')
classifier.fit(x_train, y_train)

predictions = classifier.predict(x_test)

np.savetxt('predictions.csv', predictions, fmt='%.i', delimiter=',')
