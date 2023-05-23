import numpy as np
from knn import KNN
from metrics import binary_classification_metrics, multiclass_accuracy


def crosvalidation_bin(train_X, train_y):
    k_choices = [1, 2, 3, 5, 8, 10, 15, 20, 25, 50]
    k_to_f1 = {}
    ln = len(train_X)//5

    folds_X = np.array_split(train_X, 5)
    folds_y = np.array_split(train_y, 5)

    for k in k_choices:
        f1_sum = 0
        for i in range(5):
            valid_fold_x = np.array(folds_X[i])
            valid_fold_y = np.array(folds_y[i])

            train_folds_X = np.delete(train_X, np.s_[(i * ln):(i * ln + ln)], axis=0)
            train_folds_y = np.delete(train_y, np.s_[(i * ln):(i * ln + ln)], axis=0)

            knn_classifier = KNN(k)

            knn_classifier.fit(train_folds_X, train_folds_y)
            prediction = knn_classifier.predict(valid_fold_x)

            precision, recall, f1, accuracy = binary_classification_metrics(prediction, valid_fold_y)
            f1_sum += f1
        f1_final = f1_sum/5
        k_to_f1[k] = f1_final

    for k in sorted(k_to_f1):
        print('k = %d, f1 = %f' % (k, k_to_f1[k]))


def crosvalidation_mlt(train_X, train_y):
    k_choices = [1, 2, 3, 5, 8, 10, 15, 20, 25, 50]
    k_to_accuracy = {}
    ln = len(train_X)//5

    folds_X = np.array_split(train_X, 5)
    folds_y = np.array_split(train_y, 5)

    for k in k_choices:
        accuracy_sum = 0
        for i in range(5):
            valid_fold_x = np.array(folds_X[i])
            valid_fold_y = np.array(folds_y[i])

            train_folds_X = np.delete(train_X, np.s_[(i * ln):(i * ln + ln)], axis=0)
            train_folds_y = np.delete(train_y, np.s_[(i * ln):(i * ln + ln)], axis=0)

            knn_classifier = KNN(k)

            knn_classifier.fit(train_folds_X, train_folds_y)
            prediction = knn_classifier.predict(valid_fold_x)

            accuracy = multiclass_accuracy(prediction, valid_fold_y)
            accuracy_sum += accuracy
        accuracy_final = accuracy_sum/5
        k_to_accuracy[k] = accuracy_final
    for k in sorted(k_to_accuracy):
        print('k = %d, accuracy = %f' % (k, k_to_accuracy[k]))



