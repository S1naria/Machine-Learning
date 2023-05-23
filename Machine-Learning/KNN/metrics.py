def binary_classification_metrics(prediction, ground_truth):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for i in range(len(prediction)):
        if prediction[i]:
            if ground_truth[i]:
                true_positive += 1
                continue
            false_positive += 1
            continue
        if ground_truth[i]:
            false_negative += 1
            continue
        true_negative += 1

    precision = true_positive/(true_positive+false_positive)
    recall = true_positive/(true_positive+false_negative)
    accuracy = (true_positive+true_negative)/(true_positive+true_negative+false_positive+false_negative)
    f1 = (2*precision*recall)/(precision+recall)

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    correct = 0
    for i in range(len(prediction)):
        if prediction[i] == ground_truth[i]:
            correct += 1
    return correct/len(prediction)
