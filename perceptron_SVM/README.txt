Name: Tiecheng Su
Email: tsu5@ur.rochester.edu
Course: ECE 446
Implement perceptron and SVM for the adult income dataset. Report how performance on dev varies as a function of C, and how perceptron and SVM compare.

************ Files *********
perceptron_SVM.py
README
C.png

************ Algorithm *****
Perceptron
dE/dw = sum(t_n-y_n)(dt/da)(da/dw)=sum(t_n-y_n)(d(g(a))/da)*x
a=wTx
repeat
for n=1...N do
w = w - eta*(dE/dw)
end

SVM
repeat
for n=1...N do
if 1-y(wX+b)>0
w = w-eta((1/N)*w-CyX)
b = b+eta*C*y
else
w = w -eta*(1/N)*w

************ Instructions ***
python HW5.py

************ Results ********
Perceptron
after repeat 70 times, eta=0.2
The accuracy of train dataset is 0.8427
The accuracy of test dataset is 0.8428
SVM
after repeat 20 times,eta=0.1 we can find out that we can get the highest accuracy when C equals 0.61
The accuracy of train dataset is 0.8393
The accuarcy of test dataset is 0.8481
************ Your interpretation *******
The perceptron is a linear classifier through successive changes to the slope of line according to a binary feature.
The SVM only cares about the data points near the class boundary and finds a hyperplane that maximizes the margin between classes.

************ References ************
People that I discussed with
Yanfu Zhang
Learning:Support Vector Machines MIT OpenCourseWare https://www.youtube.com/watch?v=_PwhiWxHK8o&index=1&list=LLhD_tspniD_qBTxX8gOFpGg