# Alzheimer-s-MRI-Classification-ML-Project
Dataset from: 
https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset

## Summary:
The dataset used in this project was the Alzheimer’s MRI Preprocessed Dataset obtained from Kaggle. It consisted of 6400 preprocessed 128 x 128 greyscale images of MRIs taken from Alzheimer’s patients. From the dataset, 3200 MRIs were classified under the “Normal” condition, 2240 were classified as “Very mild,” 896 were classified as “Mild,” and 64 were classified as “Moderate.” The MRIs were then randomly split into 6080 training examples and 320 test examples. Our goal was to correctly classify these MRIs into these 4 severity labels using logistic regression, support vector machines (SVMs), and fully connected neural networks. This involved creating various combinations of hyperparameters and feature transformations to obtain the highest testing set accuracy. 

For *logistic regression*, the highest testing accuracy was obtained from using L2 regularization, no feature transformations, and a C-value of either 0.01, 0.1, or 1 as they all yielded the same value of 90.625% with a training accuracy of 100%. A key characteristic of the outputted data was the significantly lower testing accuracy for the models that underwent squared and cubed feature transformations. Especially when compared to the configurations that were not transformed, this discrepancy in accuracies likely indicates that high amounts of overfitting occurred in the transformed models. Since the difference between test and training accuracies in its strongest configuration is only around 10%, this model is likely at its optimal bias and variance to minimize test error. The strong performance of this model, as well as the plateauing of improvements to test accuracies when varying Cs suggests that the only sizable improvement to this model would come from an increase in sample size. 

For *SVM*, the best result was yielded when using the RBF kernel and a regularization parameter of either 10 or 100 since the same test accuracy was obtained from both. However, the accuracies for the polynomial and RBF kernel models that used lower C-values were low for both the training and testing set. Although these are readings consistent with underfitting, we hypothesize that this may indicate underlying issues with using these models since it cannot easily classify the data under heavy regularization. A likely cause of this issue is the large discrepancy between the number of images. For example, normal images are 50 times more prominent than those classified as moderate. Possible improvements may include less strict regularization. With higher C-values, the variance is decreased and bias is increased, which allows for a relaxed hyperplane that can classify complicated data more easily. Another potential improvement may be to use a different kernel - one that may capture trends in the features more robustly. Additionally, puzzling results that spawned from using the linear kernel are constant accuracies for the training and testing sets irrespective of the penalization parameters.

For *Fully Connected Neural Network*, the configuration with the highest test accuracy used the default amount of 100 neurons in a single hidden layer while employing L2 regularization with an α-value of 1. Universally, all the neural networks tended to perform better across hidden layer structures when α was small, in other words, with less regularization applied. Additionally, the neural networks achieved better test accuracies the less hidden layers there were. This suggests that complex transformations caused too much variance in the prediction, overfitting the true model. Since the difference between test and training accuracies in its strongest configuration is only around 10% at most, this model is likely also at its optimal bias and variance to minimize test error. As the performance of the neural network more or less matches the strong performance of the logistic regression models and SVM, possible improvements would come from decreasing regularization, through decreasing α, even further, and by increasing the sample size. However, the neural network achieves differences in test and training accuracy of close to 2% in the configuration of [10,20,30] with α = 100, the strongest α we tested. This means that a potential avenue of exploration for enhancement could come from attempting increasingly complex hidden layer configurations, with heavy regularization. 

Using *PCA*, all the test accuracies of the transformed components resulted in a training accuracy of 100% and a test accuracy of 90%. In particular, using just 15 components out of the total 6080 accounted for about 40% of the original variance. This indicates that perhaps only a small handful of components are needed to reasonably represent the dataset. This could also be a result of the static nature of the positions of the images in the dataset, meaning that positions of each brain in the MRI were relatively constant. This may imply that the central components needed by the algorithms are potentially ignoring a large portion of the MRIs, perhaps being empty space surrounding the brain, as well as the outline of the brain itself. 

As a whole, the best performing model and configuration used SVM with a Gaussian RBF kernel, and a C of either 10 or 100, with a test accuracy of 93.75%. The next best performing model and configuration was a neural network with hidden layer structure [100] and α = 1, with a test accuracy of 92.1875%. Finally, the highest test accuracy of 90.625% achieved by logistic regression had a configuration using L2 norm with no feature transformation, with a C of 1, 0.1, or 0.01. 

### Works Cited
Alzheimer’s MRI Preprocessed Dataset: https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset

Extracting Images from Files: https://www.geeksforgeeks.org/how-to-use-glob-function-to-find-files-recursively-in-python/

For Logistic Regression w/ scikit-learn:
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

For SVMs w/ scikit-learn:
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

For Neural Networks w/ scikit-learn:
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

For PCA w/ scikit-learn:
http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
