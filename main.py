import itertools
from functools import reduce

import numpy as np
import pandas as pd


def entropy_method(dataset, criterion_type):
    X = np.copy(dataset) / 1.0
    for j in range(0, X.shape[1]):
        if criterion_type[j] == 'max':
            X[:, j] = X[:, j] / np.sum(X[:, j])
        else:
            X[:, j] = (1 / X[:, j]) / np.sum((1 / X[:, j]))
    X = np.abs(X)
    H = np.zeros(X.shape)
    for j, i in itertools.product(range(H.shape[1]), range(H.shape[0])):
        if X[i, j]:
            H[i, j] = X[i, j] * np.log(X[i, j])
    h = np.sum(H, axis=0) * (-1 * ((np.log(H.shape[0])) ** (-1)))
    d = 1 - h
    w = d / (np.sum(d))
    return w


def ahp_method(dataset, wd='m'):
    inc_rat = np.array([0, 0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49, 1.51, 1.48, 1.56, 1.57, 1.59])
    X = np.copy(dataset)
    weights = np.zeros(X.shape[1])
    if wd == 'm' or wd == 'mean':
        weights = np.mean(X / np.sum(X, axis=0), axis=1)
        vector = np.sum(X * weights, axis=1) / weights
        lamb_max = np.mean(vector)
    elif wd == 'g' or wd == 'geometric':
        for i in range(0, X.shape[1]):
            weights[i] = reduce((lambda x, y: x * y), X[i, :]) ** (1 / X.shape[1])
        weights = weights / np.sum(weights)
        vector = np.sum(X * weights, axis=1) / weights
        lamb_max = np.mean(vector)
    elif wd == 'me' or wd == 'max_eigen':
        eigenvalues, eigenvectors = np.linalg.eig(X)
        eigenvalues_real = np.real(eigenvalues)
        lamb_max_index = np.argmax(eigenvalues_real)
        lamb_max = eigenvalues_real[lamb_max_index]
        principal_eigenvector = np.real(eigenvectors[:, lamb_max_index])
        weights = principal_eigenvector / principal_eigenvector.sum()
    cons_ind = (lamb_max - X.shape[1]) / (X.shape[1] - 1)
    rc = cons_ind / inc_rat[X.shape[1]]
    return weights, rc


# weight_derivation = 'geometric'  # 'mean'; 'geometric' or 'max_eigen'
criterion_type = ['max']
factors = ['IF', 'MF', 'PF', 'OF']

IF_factors = ['Age', 'Gender', 'Alcoholic', 'Smoking', 'BMI', 'Low education',
              'Unhealthy diet']
MF_factors = ['Diabetes', 'Medication', 'History of stroke', 'Hypertension',
              'Cerebrovascular disease', 'Cholesterol', 'Obesity']
PF_factors = ['Anxiety', 'Depression', 'Lack of social Support', 'Loneliness',
              'Mental Fatigue', 'Sleep deprivation']
OF_factors = ['Long working hours', 'Job strain', 'Occupational attainment']

pd.set_option('display.max_columns', None)

IF_factors_dataset = pd.DataFrame(index=IF_factors)
IF_factors_dataset['Age'] = [1.000, 0.504, 2.711, 1.224, 0.607, 1.918, 1.116]
IF_factors_dataset['Gender'] = [1.986, 1.000, 1.935, 2.214, 2.104, 2.951, 2.809]
IF_factors_dataset['Alcoholic'] = [0.369, 0.517, 1.000, 0.750, 0.467, 0.491, 0.903]
IF_factors_dataset['Smoking'] = [0.817, 0.452, 1.332, 1.000, 0.325, 1.026, 1.427]
IF_factors_dataset['BMI'] = [1.647, 0.475, 2.143, 3.079, 1.000, 1.984, 1.747]
IF_factors_dataset['Low education'] = [0.521, 0.339, 2.036, 0.974, 0.504, 1.000, 0.549]
IF_factors_dataset['Unhealthy diet'] = [0.896, 0.356, 1.108, 0.701, 0.572, 1.823, 1.000]

MF_factors_dataset = pd.DataFrame(index=MF_factors)
MF_factors_dataset['Diabetes'] = [1.000, 0.851, 0.644, 0.763, 0.601, 0.296, 0.298]
MF_factors_dataset['Medication'] = [1.175, 1.000, 0.689, 0.725, 0.795, 0.408, 0.504]
MF_factors_dataset['History of stroke'] = [1.553, 1.451, 1.000, 1.301, 1.426, 1.277, 0.517]
MF_factors_dataset['Hypertension'] = [1.311, 1.380, 0.769, 1.000, 0.896, 0.372, 0.405]
MF_factors_dataset['Cerebrovascular disease'] = [1.663, 1.258, 0.701, 1.116, 1.000, 0.333, 0.530]
MF_factors_dataset['Cholesterol'] = [3.381, 2.453, 0.783, 2.690, 3.002, 1.000, 0.656]
MF_factors_dataset['Obesity'] = [3.351, 1.984, 1.935, 2.472, 1.885, 1.525, 1.000]

PF_factors_dataset = pd.DataFrame(index=PF_factors)
PF_factors_dataset['Anxiety'] = [1.000, 2.601, 0.483, 0.608, 1.392, 1.647]
PF_factors_dataset['Depression'] = [0.384, 1.000, 0.487, 0.365, 0.796, 1.034]
PF_factors_dataset['Lack of social Support'] = [2.070, 2.052, 1.000, 1.227, 2.279, 1.764]
PF_factors_dataset['Loneliness'] = [1.646, 2.737, 0.815, 1.000, 1.345, 1.246]
PF_factors_dataset['Mental Fatigue'] = [0.719, 1.256, 0.439, 0.743, 1.000, 0.823]
PF_factors_dataset['Sleep deprivation'] = [0.607, 0.967, 0.567, 0.803, 1.215, 1.000]

OF_factors_dataset = pd.DataFrame(index=OF_factors)
OF_factors_dataset['Long working hours'] = [1.000, 1.619, 0.387]
OF_factors_dataset['Job strain'] = [0.618, 1.000, 0.544]
OF_factors_dataset['Occupational attainment'] = [2.581, 1.838, 1.000]

factors_dataset = pd.DataFrame(index=factors)
factors_dataset['IF'] = [1.000, 1.402, 2.273, 1.288]
factors_dataset['MF'] = [0.713, 1.000, 1.175, 0.563]
factors_dataset['PF'] = [0.440, 0.851, 1.000, 0.782]
factors_dataset['OF'] = [0.776, 1.778, 1.278, 1.000]


def get_ranks(arr):
    sorted_weight = sorted(arr, reverse=True)
    rank = [sorted_weight.index(w) + 1 for w in arr]
    return rank


def get_factors_weight(dataset, index, weight_derivation):
    ahp_weights, ahp_rc = ahp_method(dataset, wd=weight_derivation)
    ent_weights = entropy_method(dataset, criterion_type=criterion_type * dataset.shape[0])
    dict_weight = {}
    dataset['AHP'] = ahp_weights.round(3)
    dataset['Entropy'] = ent_weights.round(3)
    print('AHP RC: ' + str(round(ahp_rc, 2)))
    if ahp_rc > 0.10:
        print('ahp_rc is bad')
    else:
        print('ahp_rc is good')
    dataset_len = dataset.shape[0]
    for i in range(dataset_len):
        numerator = ahp_weights[i] * ent_weights[i]
        denominator = 0
        for j in range(dataset_len):
            denominator += ahp_weights[j] * ent_weights[j]
        dict_weight[index[i]] = (round(numerator / denominator, 3))

    dataset['Weight'] = list(dict_weight.values())

    dataset['Rank'] = get_ranks(list(dict_weight.values()))

    print('Pairwise Comparison')
    print(dataset)
    return dict_weight


def main():
    factors_weight = get_factors_weight(factors_dataset, factors, 'geometric')
    IF_factors_weight = get_factors_weight(IF_factors_dataset, IF_factors, 'max_eigen')
    MF_factors_weight = get_factors_weight(MF_factors_dataset, MF_factors, 'max_eigen')
    PF_factors_weight = get_factors_weight(PF_factors_dataset, PF_factors, 'max_eigen')
    OF_factors_weight = get_factors_weight(OF_factors_dataset, OF_factors, 'max_eigen')

    total_weight_keys = [*IF_factors_weight.keys(), *MF_factors_weight.keys(), *PF_factors_weight.keys(),
                         *OF_factors_weight.keys()]
    for i in IF_factors_weight.keys():
        IF_factors_weight[i] *= factors_weight['IF']
    for i in MF_factors_weight.keys():
        MF_factors_weight[i] *= factors_weight['MF']
    for i in PF_factors_weight.keys():
        PF_factors_weight[i] *= factors_weight['PF']
    for i in OF_factors_weight.keys():
        OF_factors_weight[i] *= factors_weight['OF']
    total_weight_values = [*IF_factors_weight.values(), *MF_factors_weight.values(), *PF_factors_weight.values(),
                           *OF_factors_weight.values()]
    total_rank = pd.DataFrame(index=total_weight_keys)
    total_rank['Weight'] = total_weight_values
    total_rank['Rank'] = get_ranks(
        [*IF_factors_weight.values(), *MF_factors_weight.values(), *PF_factors_weight.values(),
         *OF_factors_weight.values()])
    print(total_rank)


if __name__ == "__main__":
    main()
