from numpy import genfromtxt
from sklearn.cross_validation import train_test_split


def ProcessTitanicData(CV = False):
    '''Devuelve los datos del problema del Titanic, separados en un train set, 
       un test set y un cross validation set si la opcion CV es True'''
    def convert5(x):
        '''Convierte los datos de la columna 5. los valores "male" los convierte
           en 0., y los "female" en 1.'''
        if x.strip() == 'male': ret = 0.
        elif x.strip() == 'female': ret = 1.
        else: ret = 999.
        return ret
    
    def convert6(x):
        '''Convierte los datos de la columna 6, la edad, pasando a float los datos
           existentes y poniendo el valor 0 a los datos que falten'''
        return float(x or 0)

    #Leemos los datos        
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
            X,y,Xtest,ytest = ProcessTitanicData()
            break
        elif a[0] == 's':
            X,y,XCV,yCV,Xtest,ytest = ProcessTitanicData(CV = True)
            break
        elif a[0] == 'q': break
            

