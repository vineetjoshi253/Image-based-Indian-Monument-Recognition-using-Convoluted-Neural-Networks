from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler

def preProcess(X):
    scalar=StandardScaler()
    scalar.fit(X)
    X=scalar.transform(X)
    return X

def trainTest(x_train,x_test,y_train,y_test):
    svm=SVC()
    svm.fit(x_train,y_train)
    pred=svm.predict(x_test)

    total=(len(y_test))
    c=0
    for i in range(0,total):
        if(pred[i]==y_test[i]):
            c+=1

    print(float(c)/total *100)
    
def main():
    X=np.load('Xlbp.npy')
    Y=np.load('Ylbp.npy')
    X=preProcess(X)
    print("SVM")
    x_train,x_test,y_train,y_test=train_test_split(X,Y)
    trainTest(x_train,x_test,y_train,y_test)
    

if __name__=="__main__":
    main()