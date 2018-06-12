import numpy as np
from data_utils import load_cifar10
import matplotlib.pyplot as plt
from knn import  KNearestNeighbor

x_train,y_train,x_test,y_test = load_cifar10('cifar-10-batches-py')

classes=['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
num_claesses=len(classes)
samples_per_class=7

num_training = 5000
mask = range(num_training)#(0,5000),step=1
x_train = x_train[mask] #5000*32*#2*3
y_train = y_train[mask]
num_test = 500
mask = range(num_test)
x_test = x_test[mask]
y_test = x_test[mask]

x_train = np.reshape(x_train,(x_train.shape[0],-1))
x_test = np.reshape(x_test,(x_test.shape[0],-1))

classifier = KNearestNeighbor()
classifier.train(x_train,y_train)


#比较准确率
#dists = classifier.compute_distance_two_loops(x_test)
dists = classifier.compute_distance_one_loops(x_test)
y_test_pred = classifier.predict_labels(dists,k=1)
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct)/num_test
print('get %d / %d correct =>accuracy : %f' % (num_correct ,num_test ,accuracy))

#dists_one = classifier.compute_distance_one_loops(x_test)
#difference = np.linalg.norm(dists-dists_one,ord='fro') #求范数
#print('difference was : %f' % difference)


#比较时间
import time
def time_function(f,*args):
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc- tic

two_loop_time = time_function(classifier.compute_distance_two_loops,x_test)
print('two loosp version took %f seconds' % two_loop_time)


#交叉验证
num_folds = 5
k_choices = [1,3,5,8,10,12,15,20,50,100]
x_train_folds = []
y_train_folds = []

y_train = y_train.reshape(-1,1)
x_train_folds = np.array_split(x_train,num_folds)
y_train_folds = np.array_split(y_train,num_folds)

k_to_accuracies = {}

for k in k_choices:
    k_to_accuracies.setdefault(k,[])
for i in range(num_folds):
    classifier = KNearestNeighbor()
    x_val_train = np.vstack(x_train_folds[0:i] + x_train_folds[i+1:])
    y_val_train = np.vstack(y_train_folds[0:i] + y_train_folds[i+1:])
    y_val_train = y_val_train[:,0]
    classifier.train(x_val_train,y_val_train)
    for k in k_choices:
        y_val_pred = classifier.predict_labels(x_train_folds[i],k=k)
        num_correct = np.sum(y_val_pred == y_train_folds[i][:,0])
        accuracy = float(num_correct)/len(y_val_pred)
        k_to_accuracies[k] = k_to_accuracies[k] + [accuracy]

for k in sorted(k_to_accuracies):
    sum_accuracy = 0
    for accuracy in k_to_accuracies[k]:
        print('k=%d,accuracy=%f' % (k,accuracy))
        sum_accuracy += accuracy
    print('the average accuracy is:%f' % (sum_accuracy/5))


#通过交叉验证获得最好的k值
best_k = 10
classifier = KNearestNeighbor()
classifier.train(x_train,y_train)
y_test_pred = classifier.predict(x_test,k=best_k)

num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct)/num_test
print('got %d / %d correct => accuracy:%f' % (num_correct,num_test,accuracy))