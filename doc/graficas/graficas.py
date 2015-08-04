from numpy import linspace, exp, log
from matplotlib.pyplot import plot, figure, grid, xlabel, ylabel, legend, show, savefig, imshow, subplot, cm
from sklearn.datasets import load_digits

#Tamanho fuente
fsize = 20

#Valores graficas
xsigmoid = linspace(-5,5,200)
ysigmoid = 1/(1+exp(-xsigmoid))

xerr = linspace(0,1,100)
yerr1 = -log(xerr)
yerr0 = -log(1-xerr)

xDT = linspace(0,1,100)
yent = -xDT*log(xDT) - (1-xDT)*log(1-xDT)
yGini = 2*xDT*(1-xDT)

#Graficas
#sigmoid
figure()
plot(xsigmoid, ysigmoid, linewidth=2)
grid()
xlabel('z', fontsize=fsize)
ylabel('g(z)',fontsize=fsize)

savefig('C:/Users/Javi/Desktop/TFGdefinitivo/doc/graficas/sigmoid.ps')

#LogReg
figure()
plot(xerr, yerr1, label='y = 1', linewidth=2)
plot(xerr, yerr0, label='y = 0', linewidth=2)
xlabel(r"$h_\theta(x)$", fontsize=fsize)
ylabel(r"$J(\theta)$",fontsize=fsize)
legend(loc=9,fontsize=fsize)

savefig('C:/Users/Javi/Desktop/TFGdefinitivo/doc/graficas/costLogReg.ps')

#RandFor
figure()
plot(xDT, yent, label='entropia', linewidth=2)
plot(xDT, yGini, label='Gini', linewidth=2)
xlabel(r"$p_1$", fontsize=fsize)
ylabel('criterio', fontsize = fsize)
legend(loc=8, fontsize=fsize)

savefig('C:/Users/Javi/Desktop/TFGdefinitivo/doc/graficas/DecTreeCriterios.ps')

#digitos
a = load_digits(2)
figure()
for i in range(8):
    subplot(2,4,i+1)
    imshow(a.images[i], cmap = cm.gray_r, interpolation='nearest')

savefig('C:/Users/Javi/Desktop/TFGdefinitivo/doc/graficas/digitos.svg')
    
#show()