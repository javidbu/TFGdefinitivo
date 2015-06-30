from numpy import genfromtxt
from sklearn.cross_validation import train_test_split


def ProcessTitanicData(CV = False):
    '''Devuelve los datos del problema del Titanic, separados en un train set, 
       un test set y un cross validation set si la opcion CV es True'''
    def convert5(x):
        if x.strip() == 'male': ret = 0.
        elif x.strip() == 'female': ret = 1.
        else: ret = 999.
        return ret
    
    def convert6(x):
        return float(x or 0)
        
    #COMO EN LOS NOMBRES HAY COMAS, TENGO QUE PONER UNA COLUMNA MAS
    with open('titanic/train.csv','r') as f:
        X = genfromtxt(f,skip_header = 1,
                    delimiter = ',', 
                    converters = {5: convert5, 6: convert6}, 
                    usecols = (2,5,6,7,8,10))
    
    with open('titanic/train.csv','r') as f:
        y = genfromtxt(f,skip_header = 1, 
                    delimiter = ',', 
                    usecols = (1))
        
    # LOS DATOS DEL TEST.CSV DEL KAGGLE NO TIENEN TARGET... 
    #HAY QUE HACERSE UN TEST SET DE LOS DATOS DE TRAIN.CSV
    #with open('titanic/test.csv','r') as f:
    #    Xtest = genfromtxt(f,skip_header = 1, 
    #                       delimiter = ',', 
    #                       converters = {4: convert5, 5: convert6}, 
    #                       usecols = (1,4,5,6,7,9))
    #with open('titanic/test.csv','r') as f:
    #    ytest = genfromtxt(f,skip_header = 1, 
    #                       delimiter = ',', 
    #                       usecols = (1))
        
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
            X,y,Xtest,ytest = ProcessTitanicData()
            break
        elif a[0] == 's':
            X,y,XCV,yCV,Xtest,ytest = ProcessTitanicData(CV = True)
            break
        elif a[0] == 'q': break
            

