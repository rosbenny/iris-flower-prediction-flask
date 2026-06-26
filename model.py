from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import pickle
iris=load_iris()
x=iris.data
y=iris.target
x.shape
y.shape
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
x_train_sc=scaler.fit_transform(x_train)
x_test_sc=scaler.transform(x_test)
model=LogisticRegression()
model.fit(x_train_sc,y_train)
y_pred=model.predict(x_test_sc)
print(accuracy_score(y_test,y_pred))

#Save model and scaler

pickle.dump(model,open('model.pkl','wb'))
pickle.dump(scaler,open('scalerscaler.pkl','wb'))

print('model and scaler saved successfully')