import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor

dataset = pd.read_csv('IRIS.csv')
dataset.head()
dataset.shape
dataset.describe()

#Type of data
dataset.info()

#Check for null values
dataset.isnull().sum()
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
lb = LabelEncoder()
dataset['species'] = lb.fit_transform(dataset['species'])

cols = ['sepal_length','sepal_width','petal_length','petal_width']
X = dataset.iloc[:,0:4].values
Y = dataset.iloc[:,4].values

def plot_dist( dataset , cont_features ):
    for f in cont_features :
        plt.figure(figsize=(6,3))
        sb.kdeplot(dataset[f],legend = False,color="blue",shade=True)
        plt.show()
plot_dist(dataset,cols)



#Plots 
sb.pairplot(dataset)
plt.figure(figsize=(10,11))
sb.heatmap(dataset.corr(),annot = True)
plt.plot()

#Violin Plot - density of length and width in species
plt.figure(figsize=(12,10))
plt.subplot(2,2,1)
sb.violinplot(x="species",y="sepal_length",data=dataset)
plt.subplot(2,2,2)
sb.violinplot(x="species",y="sepal_width",data=dataset)
plt.subplot(2,2,3)
sb.violinplot(x="species",y="petal_length",data=dataset)
plt.subplot(2,2,4)
sb.violinplot(x="species",y="petal_width",data=dataset)

# use boxplot to see how the categorical feature “Species” is distributed with all other four input variables.
plt.figure(figsize=(12,10))
plt.subplot(2,2,1)
sb.boxplot(x="species",y='sepal_length',data = dataset)
plt.subplot(2,2,2)
sb.boxplot(x="species",y='sepal_width',data = dataset)
plt.subplot(2,2,3)
sb.boxplot(x="species",y='petal_length',data = dataset)
plt.subplot(2,2,4)
sb.boxplot(x="species",y='petal_width',data = dataset)

from sklearn.model_selection import train_test_split
train_X , test_X , train_Y , test_Y = train_test_split(X,Y,test_size = 0.3)


#Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
dtmodel=DecisionTreeClassifier()
dtmodel.fit(train_X,train_Y)
dtpredict=dtmodel.predict(test_X)
dtup = dtmodel.predict([[7.0,2.5,5.5,1.4]])

#Check accuracy 
dtaccuracy = metrics.accuracy_score(dtpredict,test_Y)

#Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
rfModel = RandomForestClassifier(n_estimators = 100 , criterion = 'entropy' , random_state = 0 )
rfModel.fit(train_X,train_Y)
rfPred = rfModel.predict(test_X)
rfaccuracy = metrics.accuracy_score(rfPred,test_Y)


#Neural Network
model = Sequential()
model.add(Dense(12, input_dim=4, kernel_initializer='normal', activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
Mod=model.fit(train_X,train_Y, epochs=50, batch_size=50,  verbose=1, validation_split=0.2)

print(Mod.history.keys())
plt.plot(Mod.history['loss'])
plt.plot(Mod.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

y_pred = model.predict(test_X)