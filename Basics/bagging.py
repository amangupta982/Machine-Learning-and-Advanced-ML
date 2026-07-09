from sklearn.ensemble import BaggingClassifier 
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset = load_iris()
X = dataset.data
y= dataset.target

X_train , X_test , y_train , y_test = train_test_split(X,y, test_size = 0.3 , random_state = 42)

bagging = BaggingClassifier(DecisionTreeClassifier(),n_estimators = 10 , random_state = 42)
bagging.fit(X_train , y_train)

y_pred = bagging.predict(X_test)

print ("Accuray_score is :",accuracy_score(y_test , y_pred))
