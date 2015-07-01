from numpy import linspace, exp, log
from matplotlib.pyplot import plot, figure, grid, xlabel, ylabel, legend, show, savefig

fsize = 20

xsigmoid = linspace(-5,5,200)
ysigmoid = 1/(1+exp(-xsigmoid))

xerr = linspace(0,1,100)
yerr1 = -log(xerr)
yerr0 = -log(1-xerr)

figure()
plot(xsigmoid, ysigmoid, linewidth=2)
grid()
xlabel('z', fontsize=fsize)
ylabel('g(z)',fontsize=fsize)

savefig('C:/Users/Javi/Desktop/TFGdefinitivo/doc/graficas/sigmoid.ps')

figure()
plot(xerr, yerr1, label='y = 1', linewidth=2)
plot(xerr, yerr0, label='y = 0', linewidth=2)
xlabel(r"$h_\theta(x)$", fontsize=fsize)
ylabel(r"$J(\theta)$",fontsize=fsize)
legend(loc=9,fontsize=fsize)

savefig('C:/Users/Javi/Desktop/TFGdefinitivo/doc/graficas/costLogReg.ps')

#show()