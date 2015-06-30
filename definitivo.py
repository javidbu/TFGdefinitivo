#Metricas
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
#Modelos
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
#Cross Validation
from sklearn.grid_search import GridSearchCV
#Preprocesamiento de datos
from sklearn.datasets import load_digits
from PreProcTitanic import ProcessTitanicData
from sklearn.preprocessing import MinMaxScaler
#Graficas
from matplotlib.pyplot import figure, scatter, show, contourf, title, subplot
#numpy-cosas
from numpy import meshgrid, arange, c_

class Clasificadores():
    def __init__(self, X, y, Xtest, ytest, FeatScal = True):
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
	self.clasificadores = [SVC(C=1, kernel = 'rbf'), 
	                       LogisticRegression(C=1), 
	                       GaussianNB(), #Este puede usar partial_fit para ajustar a datos nuevos... Interesante cuando haya un monton de ellos...
	                       RandomForestClassifier(n_estimators = 10, criterion = 'gini',max_depth = None, min_samples_split = 2, min_samples_leaf = 1), 
	                       KNeighborsClassifier(n_neighbors = 5)]
	#Cuando consiga hacer el gridSearch, hay que cambiar cada clasificador por los clf.best_estimator_ (salvo el GaussianNB)
	self.param = ([[{'kernel': ['rbf'], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1.0], 'C': [1, 10, 100, 1000]},#Esto es para hacer el crossValidation...
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}],#SVM
	              [{'C': [1, 10, 100, 1000]}],#LogReg
	              [{'n_estimators': [5, 10, 15], 'min_samples_split': [2, 4, 6, 8, 10], 'min_samples_leaf': [1, 2, 3, 4, 5]}],#RandFor
	              [{'n_neighbors': [2, 3, 4, 5, 6]}]])#KNeighb
	self.grid = [GridSearchCV(SVC(C=1),self.param[0],cv=5, scoring='f1_weighted'),
	             GridSearchCV(LogisticRegression(C=1),self.param[1],cv=5, scoring='f1_weighted'),
	             GridSearchCV(RandomForestClassifier(n_estimators=10),self.param[2],cv=5, scoring='f1_weighted'),
	             GridSearchCV(KNeighborsClassifier(n_neighbors=5),self.param[3],cv=5, scoring='f1_weighted')]
	self.predicciones = {}
	self.acc = {}
	self.prec = {}
	self.rec = {}
	self.f1 = {}
	self.fitted = False
	self.predicted = False
	self.newClassifiers = [GaussianNB().fit(self.X,self.y)]

    def fit(self):
	print '#'*50
	for c in self.clasificadores:
	    print 'Fitting %s' % c.__class__.__name__
	    c.fit(self.X,self.y)
	self.fitted = True

    def fit2(self):
        for g in self.grid:
            g.fit(self.X, self.y)
            self.newClassifiers.append(g.best_estimator_)
        self.fitted = True
            
    def predict(self):
        if not self.fitted: self.fit()
        print '#'*50
        for c in self.clasificadores:
            print 'Predicting %s' % c.__class__.__name__
            self.predicciones[str(c).split('(')[0]] = c.predict(self.Xtest)
        self.predicted = True
             
    def predict2(self):
        if not self.fitted: self.fit2()
        print '#'*50
        for c in self.newClassifiers:
            print 'Predicting %s' % c.__class__.__name__
            self.predicciones[str(c).split('(')[0]] = c.predict(self.Xtest)
        self.predicted = True
            
    def metricas(self):
        if not self.predicted: self.predict()
	for c in self.clasificadores:
	    nombre = c.__class__.__name__
	    self.acc[nombre] = accuracy_score(self.ytest, self.predicciones[nombre])
	    self.prec[nombre] = precision_score(self.ytest, self.predicciones[nombre])
	    self.rec[nombre] = recall_score(self.ytest, self.predicciones[nombre])
	    self.f1[nombre] = f1_score(self.ytest, self.predicciones[nombre])
	    print '#'*50
	    print 'Metricas para %s' % nombre
	    print 'acc\t%3.4f\nprec\t%3.4f\nrec\t%3.4f\nf1\t%3.4f' % (self.acc[nombre],self.prec[nombre],self.rec[nombre],self.f1[nombre])
    
    def metricas2(self):
        if not self.predicted: self.predict2()
	for c in self.newClassifiers:
	    nombre = c.__class__.__name__
	    self.acc[nombre] = accuracy_score(self.ytest, self.predicciones[nombre])
	    self.prec[nombre] = precision_score(self.ytest, self.predicciones[nombre])
	    self.rec[nombre] = recall_score(self.ytest, self.predicciones[nombre])
	    self.f1[nombre] = f1_score(self.ytest, self.predicciones[nombre])
	    print '#'*50
	    print 'Metricas para %s' % nombre
	    print 'acc\t%3.4f\nprec\t%3.4f\nrec\t%3.4f\nf1\t%3.4f' % (self.acc[nombre],self.prec[nombre],self.rec[nombre],self.f1[nombre])
    
    def graficas(self):
        if not self.fitted: self.fit()
        x_min, x_max = self.X[:,0].min(), self.X[:,0].max()
        y_min, y_max = self.X[:,1].min(), self.X[:,1].max()
        hx = (x_max-x_min)/100. #Paso para el grid
        hy = (y_max-y_min)/100.
        x_min, x_max,y_min,y_max = x_min - hx, x_max + hx, y_min - hy, y_max + hy
        xx, yy = meshgrid(arange(x_min, x_max, hx),arange(y_min, y_max, hy))
        figure()
        i = 1
        subplot(2,3,i)
        scatter(self.X[:,0],self.X[:,1],c = self.y, s = 100)
        scatter(self.Xtest[:,0],self.Xtest[:,1],c = self.ytest, s = 100, alpha = 0.6)
        title('Datos')
        for c in self.clasificadores:
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

    def graficas2(self):
        if not self.fitted: self.fit2()
        x_min, x_max = self.X[:,0].min(), self.X[:,0].max()
        y_min, y_max = self.X[:,1].min(), self.X[:,1].max()
        hx = (x_max-x_min)/100. #Paso para el grid
        hy = (y_max-y_min)/100.
        x_min, x_max,y_min,y_max = x_min - hx, x_max + hx, y_min - hy, y_max + hy
        xx, yy = meshgrid(arange(x_min, x_max, hx),arange(y_min, y_max, hy))
        figure()
        i = 1
        subplot(2,3,i)
        scatter(self.X[:,0],self.X[:,1],c = self.y, s = 100)
        scatter(self.Xtest[:,0],self.Xtest[:,1],c = self.ytest, s = 100, alpha = 0.6)
        title('Datos')
        for c in self.newClassifiers:
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


def Digitos():
    columnas = (20,27)
    print 'Usando los datos de los digitos'
    a = load_digits(2)
    X = a['data'][:300,columnas]
    Xtest = a['data'][300:,columnas]
    y = a['target'][:300,]
    ytest = a['target'][300:,]
    c = Clasificadores(X,y,Xtest,ytest)
    c.metricas()
    c.graficas()
    #print c.clasificadores[3].feature_importances_
    return c
    
def Digitos2():
    columnas = (20,27)
    print 'Usando los datos de los digitos'
    a = load_digits(2)
    X = a['data'][:300,columnas]
    Xtest = a['data'][300:,columnas]
    y = a['target'][:300,]
    ytest = a['target'][300:,]
    c = Clasificadores(X,y,Xtest,ytest)
    c.metricas2()
    c.graficas2()
    #print c.clasificadores[3].feature_importances_
    return c
    
def Titanic():
    columnas = (1,5)
    print 'Usando los datos del Titanic'
    X,y,Xtest,ytest = ProcessTitanicData()
    X = X[:,columnas]
    Xtest = Xtest[:,columnas]
    c = Clasificadores(X,y,Xtest,ytest)
    c.metricas()
    c.graficas() 
    #print c.clasificadores[3].feature_importances_
    return c
    return c
    
def Titanic2():
    columnas = (2,5)
    print 'Usando los datos del Titanic'
    X,y,Xtest,ytest = ProcessTitanicData()
    X = X#[:,columnas]
    Xtest = Xtest#[:,columnas]
    c = Clasificadores(X,y,Xtest,ytest)
    c.metricas2()
    c.graficas2()
    #print c.clasificadores[3].feature_importances_
    return c

if __name__ == '__main__':
    #c = Digitos()
    #c = Titanic()
    #c = Digitos2()
    c = Titanic2()