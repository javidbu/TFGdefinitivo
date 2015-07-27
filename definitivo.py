#Metricas
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
#Modelos
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB #Igual nos interesa tambien MultinomialNB...
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
#Cross Validation
from sklearn.grid_search import GridSearchCV
#Curvas de aprendizaje
from sklearn.learning_curve import learning_curve
#Preprocesamiento de datos
from sklearn.datasets import load_digits
from PreProcTitanic import ProcessTitanicData
from PreProcHiggs import ProcessHiggsData
from sklearn.preprocessing import MinMaxScaler
#Graficas
from matplotlib.pyplot import figure, scatter, show, contourf, title, subplot, xlabel, ylabel, fill_between, legend, grid, plot, ylim
#numpy-cosas
from numpy import meshgrid, arange, c_, linspace, mean, std

class Clasificadores():
    '''Clase para llevar a cabo los diversos algoritmos de clasificacion, asi como
       el Cross Validation, Feature Scaling...'''
    def __init__(self, X, y, Xtest, ytest, FeatScal = True, CV = True, CurvaAprendizaje = False, datos = ''):
        '''X, y, Xtest, ytest son los datos. FeatScal es True si se quiere hacer
           Feature Scaling (True por defecto). CV es True si se quiere hacer el 
           Cross Validation (True por defecto)'''
        self.CV = CV
        self.CurvAp = CurvaAprendizaje
        self.datos = datos
        if FeatScal:
            self.FeatScal = MinMaxScaler(copy=False)
            #Para deshacer el featScal: self.FeatScal.inverse_transform(X)
            self.X = self.FeatScal.fit_transform(X) # Hace feature scaling a la X
            self.Xtest = self.FeatScal.transform(Xtest)
        else:
            self.X = X
            self.Xtest = Xtest
	self.y = y
	self.ytest = ytest
	if self.CV:
	   self.param = ([[{'kernel': ['rbf'], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1.0], 'C': [1, 10, 100, 1000]},#Esto es para hacer el crossValidation...
                           {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}],#SVM
	                   [{'C': [1, 10, 100, 1000]}],#LogReg
	                   [{'n_estimators': [5, 10, 15], 'min_samples_split': [2, 4, 6, 8, 10], 'min_samples_leaf': [1, 2, 3, 4, 5]}],#RandFor
	                   [{'n_neighbors': [2, 3, 4, 5, 6]}]])#KNeighb
	   self.grid = [GridSearchCV(SVC(C=1),self.param[0],cv=5, scoring='f1_weighted'),
	                GridSearchCV(LogisticRegression(C=1),self.param[1],cv=5, scoring='f1_weighted'),
	                GridSearchCV(RandomForestClassifier(n_estimators=10),self.param[2],cv=5, scoring='f1_weighted'),
	                GridSearchCV(KNeighborsClassifier(n_neighbors=5),self.param[3],cv=5, scoring='f1_weighted')]
	   self.newClassifiers = [GaussianNB().fit(self.X,self.y)]#Esto es asi porque el NaiveBayes no tiene parametros para el CV
        else:
	   self.clasificadores = [SVC(C=1, kernel = 'rbf'), 
	                          LogisticRegression(C=1), 
	                          GaussianNB(), #Este puede usar partial_fit para ajustar a datos nuevos... Interesante cuando haya un monton de ellos...
	                          RandomForestClassifier(n_estimators = 10, criterion = 'gini',max_depth = None, min_samples_split = 2, min_samples_leaf = 1), 
	                          KNeighborsClassifier(n_neighbors = 5)]
	self.learningCurve = []
	self.predicciones = {}
	self.acc = {}
	self.prec = {}
	self.rec = {}
	self.f1 = {}
	self.fitted = False
	self.predicted = False

    def fit(self):
        '''Si self.CV es True, hace el cross validation y guarda en newClassifiers 
           el mejor clasificador para cada algoritmo. Si no, hace el fit de los
           clasificadores'''
        if self.CV:
            for g in self.grid:
                g.fit(self.X, self.y)
                self.newClassifiers.append(g.best_estimator_)
            self.fitted = True
        else:
	   print '#'*50
	   for c in self.clasificadores:
	       print 'Fitting %s' % c.__class__.__name__
	       c.fit(self.X,self.y)
	   self.fitted = True
        
    def predict(self):
        '''Predice los valores del test set para los clasificadores entrenados'''
        if not self.fitted: self.fit()
        if self.CV:
            a = self.newClassifiers
        else:
            a = self.clasificadores
        print '#'*50
        for c in a:
            print 'Predicting %s' % c.__class__.__name__
            self.predicciones[str(c).split('(')[0]] = c.predict(self.Xtest)
        self.predicted = True
             
    def metricas(self):
        '''Calcula las diferentes metricas para los clasificadores entrenados y
           predichos, las guarda en los correspondientes diccionarios y las im-
           prime en pantalla'''
        if not self.predicted: self.predict()
        if self.CV:
            a = self.newClassifiers
        else:
            a = self.clasificadores
        for c in a:
            nombre = c.__class__.__name__
            self.acc[nombre] = accuracy_score(self.ytest, self.predicciones[nombre])
            self.prec[nombre] = precision_score(self.ytest, self.predicciones[nombre])
            self.rec[nombre] = recall_score(self.ytest, self.predicciones[nombre])
            self.f1[nombre] = f1_score(self.ytest, self.predicciones[nombre])
            print '#'*50
            print 'Metricas para %s' % nombre
            print 'acc\t%3.4f\nprec\t%3.4f\nrec\t%3.4f\nf1\t%3.4f' % (self.acc[nombre],self.prec[nombre],self.rec[nombre],self.f1[nombre])
    
    def graficas(self):
        '''Si los datos tienen solo dos features, hace las graficas para cada 
           clasificador'''
        if not self.fitted: self.fit()
        try:
            if self.CV:
                a = self.newClassifiers
            else:
                a = self.clasificadores
            x_min, x_max = self.X[:,0].min(), self.X[:,0].max()
            y_min, y_max = self.X[:,1].min(), self.X[:,1].max()
            hx = (x_max-x_min)/100. #Paso para el grid
            hy = (y_max-y_min)/100.
            x_min, x_max,y_min,y_max = x_min - hx, x_max + hx, y_min - hy, y_max + hy
            xx, yy = meshgrid(arange(x_min, x_max, hx),arange(y_min, y_max, hy))
            figure(self.datos + ', curvas de decision')
            i = 1
            subplot(2,3,i)
            title('Datos')
            scatter(self.X[:,0],self.X[:,1],c = self.y, s = 100)
            scatter(self.Xtest[:,0],self.Xtest[:,1],c = self.ytest, s = 100, alpha = 0.6)
            for c in a:
                i += 1
                subplot(2,3,i)
                Z = c.predict(c_[xx.ravel(),yy.ravel()])
                #if hasattr(c,'decision_function'):
                #    Z = c.decision_function(c_[xx.ravel(),yy.ravel()])
                #else:
                #    Z = c.predict_proba(c_[xx.ravel(),yy.ravel()])[:,1]
                Z = Z.reshape(xx.shape)
                contourf(xx,yy,Z,alpha = 0.8)
                scatter(self.X[:,0],self.X[:,1],c = self.y, s = 100)
                scatter(self.Xtest[:,0],self.Xtest[:,1],c = self.ytest, s = 100, alpha = 0.6)
                title(c.__class__.__name__)
            show()
        except ValueError:
            print 'No se pudieron hacer las graficas'
    
    def CurvaAprendizaje(self):
        if self.CV:
            if not self.fitted: self.fit()
            a = self.newClassifiers
        else:
            a = self.clasificadores
        figure(self.datos + ', curvas de aprendizaje')
        i=1
        for c in a:
            if c.__class__.__name__ == 'GaussianNB': 
                modo = True
            else:
                modo = False
            numero, training, CV = learning_curve(c, self.X, self.y, 
                                                    train_sizes = linspace(0.1,1.,10), 
                                                    cv = 5, 
                                                    scoring = 'f1_weighted',
                                                    exploit_incremental_learning = modo
                                                    )
            subplot(2,3,i)
            ylim((0,1))
            title(c.__class__.__name__)
            xlabel('# de datos')
            ylabel('F1')
            training_mean, training_std = mean(training, axis=1), std(training, axis=1)
            CV_mean, CV_std = mean(CV, axis=1), std(CV, axis=1)
            grid()
            fill_between(numero, training_mean - training_std, training_mean + training_std, color = 'r', alpha = 0.1)
            fill_between(numero, CV_mean - CV_std, CV_mean + CV_std, color = 'g', alpha = 0.1)
            plot(numero, training_mean, 'o-', color='r', label = 'Training')
            plot(numero, CV_mean, 'o-', color='g', label = 'Cross Validation')
            legend(loc = 4)
            i += 1
        show()
            
    
    def todo(self):
        '''Lo hace todo para los clasificadores'''
        self.metricas()
        self.graficas()
        if self.CurvAp:
            self.CurvaAprendizaje()

def Digitos(cv = True,CAp = False):
    '''Prepara los datos de los digitos y hace las clasificaciones.'''
    columnas = (20,27)
    print 'Usando los datos de los digitos'
    a = load_digits(2)
    X = a['data'][:300,columnas]
    Xtest = a['data'][300:,columnas]
    y = a['target'][:300,]
    ytest = a['target'][300:,]
    c = Clasificadores(X,y,Xtest,ytest, CV = cv, CurvaAprendizaje = CAp, datos = 'Digitos')
    c.todo()
    #print c.clasificadores[3].feature_importances_
    return c
    
def Titanic(cv = True,CAp = False):
    '''Prepara los datos del Titanic y hace las clasificaciones.'''
    columnas = (1,5)
    print 'Usando los datos del Titanic'
    X,y,Xtest,ytest = ProcessTitanicData()
    X = X[:,columnas]
    Xtest = Xtest[:,columnas]
    c = Clasificadores(X,y,Xtest,ytest, CV = cv, CurvaAprendizaje = CAp, datos = 'Titanic')
    c.todo()
    #print c.clasificadores[3].feature_importances_
    return c

def Higgs(cv = True, CAp = False):
    '''Prepara los datos del Titanic y hace las clasificaciones.'''
    #columnas = (,)
    print 'Usando los datos del Higgs'
    X, y, Xtest, ytest = ProcessHiggsData()
    #X = X[:,columnas]
    #Xtest = Xtest[:,columnas]
    c = Clasificadores(X, y, Xtest, ytest, CV = cv, CurvaAprendizaje = CAp, datos = 'Higgs')
    c.todo()
    print c.clasificadores[3].feature_importances_
    
if __name__ == '__main__':
    #d = Digitos(CAp = True)
    #t = Titanic(CAp = True)
    h = Higgs(False, False)#No poner nunca True para la curva de aprendizaje hasta el final, tarda la del pulpo morado...