import pandas as pd
import random
import math

def load_dataset(filename):
    return pd.read_csv(filename)

def split_dataset(dataset, split_ratio):
    num_rows = len(dataset)
    train_size = int(num_rows * split_ratio)
    indices = list(range(num_rows))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    train_set = dataset.iloc[train_indices].values.tolist()
    test_set = dataset.iloc[test_indices].values.tolist()
    return train_set, test_set

def separate_by_class(dataset):
    separated = {}
    for row in dataset:
        if row[-1] not in separated:
            separated[row[-1]] = []
        separated[row[-1]].append(row)
    return separated

def mean(numbers):
    return sum(numbers) / float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)

def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = {}
    for class_value, instances in separated.items():
        summaries[class_value] = summarize(instances)
    return summaries

def calculate_probability(x, mean, stdev):
    exponent = math.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

def calculate_class_probabilities(summaries, input_vector):
    probabilities = {}
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = 1
        for i in range(len(class_summaries)):
            mean, stdev = class_summaries[i]
            x = input_vector[i]
            probabilities[class_value] *= calculate_probability(x, mean, stdev)
    return probabilities

def predict(summaries, input_vector):
    probabilities = calculate_class_probabilities(summaries, input_vector)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label

def get_predictions(summaries, test_set):
    predictions = []
    for i in range(len(test_set)):
        result = predict(summaries, test_set[i])
        predictions.append(result)
    return predictions

def get_accuracy(test_set, predictions):
    correct = 0
    for i in range(len(test_set)):
        if test_set[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(test_set))) * 100.0

filename = 'Breast_cancer_data.csv'
dataset = load_dataset(filename)

split_ratio = 0.8
train_set, test_set = split_dataset(dataset, split_ratio)

summaries = summarize_by_class(train_set)

predictions = get_predictions(summaries, test_set)

accuracy = get_accuracy(test_set, predictions)
print('Accuracy:', accuracy)




print("Test set and Predictions:")
for i in range(len(test_set)):
    print("Test instance:", test_set[i], "\tPredicted class:", predictions[i])