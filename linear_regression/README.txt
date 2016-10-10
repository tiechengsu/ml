Name: Tiecheng Su
Email: tsu5@ur.rochester.edu
Course: ECE446
Homework: Implement linear regression for the adult income dataset using Python. Data available in /u/cs246/data/adult and /u/cs446/data/adult. Take the sign of the output of linear regression as the predicted class label in order to compute the accuracy of your classifier.

************ Files *********
linear_regression.py
README

************ Algorithm *****
w = inverse(transpose(X)*X)*transpose(X)*y
w = inverse(transpose(X)*X+lambda*I)*transpose(X)*y

************ Instructions ***
python HW2.py extra_test_file_name

************ Results *******
The accuracy of dev dataset is 0.84925 and the accuracy of test data is 0.84364
Parameter selection procedure on the dev set
Just simply try different lambdas and I find out that the classification accuracy is the best when the value of lambda is between 0 and 1.8

************ Your interpretation *******
The accuracy of the three sets are all higher than 0.84, which is pretty good.
First, we need to use w = inverse(transpose(X)*X)*transpose(X)*y to calculate the weight of train set. And then try different lambdas base on the dev set using w = inverse(transpose(X)*X+lambda*I)*transpose(X)*y. Since we only need to take the sign of the output as the predicted class label, we can simply use np.copysign to process the output data. Finally, calculate the difference between the processed data and the binary class.

************ References ************
People who I discussed with
Xiangru lian
Chang Ye
Hao Xie
Yanfu Zhang