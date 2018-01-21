
# pip install numpy scipy matplotlib ipython scikit-learn pandas
import  numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
#import scipy as sp
import mglearn # get it here "https://github.com/amueller/mglearn" and install it via python setup.py install
#from IPython import display

from sklearn.datasets import  load_iris
iris_dataset = load_iris()



#Explore iris_dataset
print ("Schlüssel von iris_dataset: \n{}".format(iris_dataset.keys()))
print ("Beschreibungstext:")
print (iris_dataset["DESCR"][:])
print ("Zielbezeichnungen: {}".format(iris_dataset["target_names"]))
print ("Name der Merkmale: \n".format(iris_dataset["feature_names"]))
print("Typ der Daten: {}".format(type(iris_dataset["data"])))
print ("Abmessung der Daten: {}".format(iris_dataset["data"].shape))
print("Die ersten 10 Zeilen der Daten:\n{}".format(iris_dataset["data"][:10]))
print("Typ der Zielgoesse: {}".format(type(iris_dataset["target"])))
print ("Abmessung der Zielgroesse: {}".format(iris_dataset["target"].shape))
print("Zielwerte:\n{}".format(iris_dataset["target"][:]))



#Create Train and Test Datasets:

from sklearn.model_selection import train_test_split

#Datensatz der Merkmale und Zielwerte zufällig
#in einen TrainingsMerkmaldatensatz und einen entsprechenden TrainingsZielwertdatensatz (75%) --> X
#und einen TestMerkmaldatensatz und einen entsprechenden TestZielwertdatensatz (25%)  aufteilen --> y
X_train,X_test, y_train, y_test = train_test_split(iris_dataset["data"], iris_dataset["target"], random_state=0)
print ("Abmessung von X_train: {}".format(X_train.shape))
print ("Abmessung von X_test: {}".format(X_test.shape))
print ("Abmessung von y_train: {}".format(y_train.shape))
print ("Abmessung von y_test: {}".format(y_test.shape))





#Traingsdaten visuell darstellen um sie zu inspzieren: sind die drei ZielKategroien gut durch messung von Kelch und Kronblättern voneinander abgegrenzt?
# Wenn das der Fall ist, dann steigt die Wahrscheinlichkeit, dass das maschinelle Lernmodel in der lage sein wird, diese Kategorien zu unterscheiden.

#1.erstelle aus den Daten in X_train ein PandasDataFrame
#Verwende die Strings aus iris_dataset.feature_names als Spaltenüberschriften.
iris_dataframe_X_train = pd.DataFrame(X_train, columns= iris_dataset.feature_names)
print (iris_dataframe_X_train)
#erstelle eine Matrix von Streudiagrammen aus dem DataFrame
#färbe nach y_train ein
grr= pd.plotting.scatter_matrix(iris_dataframe_X_train, c=y_train, figsize= (15, 15), marker="o", hist_kwds= {"bins":20}, s=60, alpha=.8, cmap=mglearn.cm3)
plt.show()





#Konstruktion (einlernen?) des Modells:
from sklearn .neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors =1)
knn.fit(X_train, y_train)#Modell aus den Traingsdatensätzen aufbauen
print(knn)





#vorhersagen treffen
#Testobjekt generieren
X_new = np.array([[5, 2.9, 1, 0.2]])
print ("X_new.shape: {}".format(X_new.shape))
#Vorhersage für dieses Testobjekt treffen
y_prediction = knn.predict(X_new)
print ("Vorhersage: {}".format(y_prediction))
print("Vorhergesagter Name: {}".format(iris_dataset["target_names"][y_prediction]))


#Evaluation der Genauigkeit des Modells.
y_test_prediction = knn.predict(X_test)
print ("Vorhersage für den Testdatensatz:\n {}".format(y_test_prediction))
print ("Tatsächliche Werte im Testdatensatz:\n {}".format(y_test))
print("Genauigkeit auf den Testdaten: {:.2f}".format(np.mean(y_test_prediction == y_test)))
print("Genauigkeit auf den Testdaten: {:.2f} (alternativ direktüber knn objekt berechnet)".format(knn.score(X_test,y_test)))

