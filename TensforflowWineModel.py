import numpy as np
import tensorflow as tf 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as t

file = 'winequality-red.csv'
DATA = pd.read_csv(file)


print(DATA.shape)
print(DATA.head)
DATA.info()


# pairplot = sns.pairplot(DATA, hue='quality')
# pairplot.savefig('pairplot.png')

mask = np.zeros_like(DATA.corr())
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(14,6))
sns.heatmap(DATA.corr(), cmap = 'viridis', mask=mask, annot=False, square=True)

(abs(DATA.corr()[['fixed acidity', 'volatile acidity', 'total sulfur dioxide']])> .6) * 1

data = DATA.drop(columns=['citric acid', 'density', 'pH', 'total sulfur dioxide'])
data.head()

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

X = data.loc[:, data.columns != 'quality'].values
y = data.quality.values
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=40)

y_train_cat = to_categorical(y_train, 6)
y_test_cat = to_categorical(y_test, 6)


scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train.shape)
print(X_test.shape)

ben_init = tf.keras.initializers.GlorotNormal()

model = Sequential()
model.add(Dense(48, kernel_initializer=ben_init,  activation='relu'))
model.add(Dense(24, kernel_initializer=ben_init, activation='relu'))
model.add(Dense(12, kernel_initializer=ben_init, activation='relu'))
model.add(Dense(6, kernel_initializer=ben_init,
activation='relu'))
model.add(Dense(6, kernel_initializer=ben_init, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train,
y_train_cat, epochs=22,validation_data=(X_test, y_test_cat), verbose=0)


losses = pd.DataFrame(model.history.history)

losses[['loss','val_loss']].plot()
losses[['accuracy','val_accuracy']].plot()

print(model.metrics_names)
print(model.evaluate(X_test,y_test_cat,verbose=0))

from tensorflow.keras.models import save_model

save_model(model, 'wine_model')