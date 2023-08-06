import sklearn
from sklearn import svm
from sklearn import preprocessing
import pandas as pd
from sklearn import metrics

le = preprocessing.LabelEncoder()
datas = pd.read_csv("Iris.csv")
data = datas[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm",
              "PetalWidthCm"]]
predict = le.fit_transform(list(datas["Species"]))

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data, predict, test_size=0.1)

SVM = svm.SVC(kernel="linear", C=3)
SVM.fit(x_train, y_train)

prediction = SVM.predict(x_test)
acc = metrics.accuracy_score(y_test, prediction)
print(acc)
