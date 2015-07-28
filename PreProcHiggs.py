from numpy import genfromtxt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer

def ProcessHiggsData(CV = False):
    '''Devuelve los datos del problema del Higgs, separados en un train set,
       un test set y un cross validation set si la opcion CV es True'''
    def convert(x):
        if x == 's': return 0
        elif x == 'b': return 1
        return 999
    
    #Leemos los datos
    with open('higgs/training.csv','r') as f:
        X = genfromtxt(f, skip_header = 1,
                       delimiter = ',',
                       usecols = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30))
    
    with open('higgs/training.csv','r') as f:
        y = genfromtxt(f, skip_header = 1,
                       delimiter = ',',
                       converters = {32: convert},
                       usecols = (32))
    
    #Reemplazamos los missing values por la media de cada feature
    i = Imputer(missing_values = -999.0,
                strategy = 'mean',
                axis = 0)
    X = i.fit_transform(X)
    
    
    #Separamos los datos en los diferentes sets
    if not CV:
        train_perct = 0.8
        X,Xtest,y,ytest = train_test_split(X,y,train_size = train_perct)
        return X,y,Xtest,ytest
    else:
        train_perct = 0.6
        X,Xprov,y,yprov = train_test_split(X,y,train_size = train_perct)
        XCV,Xtest,yCV,ytest = train_test_split(Xprov,yprov,train_size = 0.5)
        return X,y,XCV,yCV,Xtest,ytest
    

if __name__ == '__main__':
    while True:
        a = raw_input('Quiere un CV set? ([s]i/[n]o/[q]=salir)\n> ').lower()
        if a[0] == 'n':
            X,y,Xtest,ytest = ProcessHiggsData()
            break
        elif a[0] == 's':
            X,y,XCV,yCV,Xtest,ytest = ProcessHiggsData(CV = True)
            break
        elif a[0] == 'q': break