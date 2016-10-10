Name: Tiecheng Su
Email: tsu5@ur.rochester.edu
Course: ECE 446
Homework: Implement naive bayes with Dirichlet smoothing for the adult income dataset. Plot classification error on the dev set as a function of alpha, 
and report the performance of the best alpha on the test set.

************ Files *********
naive_bayes.py
README
dev_set.png

************ Algorithm *****
Naive Bayes
P(y|x1,...xn)=P(y)P(xi|y)
Dirichlet smoothing
P(x=k|y)=(c(x=k,y)+alpha)/(c(y)+K*alpha)

************ Instructions ***
python HW3.py

************ Results *******
Calculating the classification error with different alpha. We can find out that when alpha equals to 308, the classification error reach its minimum, 
which is 0.1667. The classification error of test set is 0.1684

************ Your interpretation *******
First, we need to calculate P(x|y) base on train set. As we know, some unseen combination of a single feature x and the class label y can result in 
P(x|y)=0, therefore we use Dirichlet smoothing to modify the probabilities base on dev set. According to the experiment, we can get the best result 
when alpha equals 308. At last, we use the best alpha to calculate the classification error of test set.

************ References ************
People that I discussed with
Yanfu Zhang