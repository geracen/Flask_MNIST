import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from util import get_data
from sklearn import metrics

if __name__=='__main__':
    X,Y=get_data()
    Ntrain=len(Y)//4

    Xtrain, Ytrain=X[:Ntrain],Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]
    model=RandomForestClassifier()
    model.fit(Xtrain,Ytrain)

    with open('mymodel.pkl','wb') as f:
        pickle.dump(model,f)


