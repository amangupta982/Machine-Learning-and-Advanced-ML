from catboost import CatBoostClassifier
from sklearn.datasets import load_iris 
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split

dataset = load_iris()
X = dataset.data
y = dataset.target 

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.25, random_state = 42)

cat = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=5, verbose=0)
cat.fit(X_train,y_train)

y_pred = cat.predict(X_test)

print ("Acccuracy Score :" , accuracy_score(y_pred , y_test ))