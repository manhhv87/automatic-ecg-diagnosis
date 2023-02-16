# %% Import packages
import pandas as pd
import numpy as np
from sklearn.metrics import (precision_score, recall_score, f1_score, average_precision_score)
from utils import (specificity_score, generate_table, plot_pre_rec_curve, plot_confusion_matrix,
                   compute_score_bootstraped, plot_box, McNemar_score, kappa_score_classifier,
                   kappa_score_dataset_generation, compute_score_bootstraped_splits, plot_box_splits)

# %% Constants
score_fun = {'Precision': precision_score, 'Recall': recall_score,
             'Specificity': specificity_score, 'F1 score': f1_score}
diagnosis = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST']
nclasses = len(diagnosis)
predictor_names = ['DNN', 'cardio.', 'emerg.', 'stud.']

# %% Read datasets
# Get two annotators
y_cardiologist1 = pd.read_csv('./data/annotations/cardiologist1.csv').values
y_cardiologist2 = pd.read_csv('./data/annotations/cardiologist2.csv').values

# Get residents and students performance
y_cardio = pd.read_csv('./data/annotations/cardiology_residents.csv').values
y_emerg = pd.read_csv('./data/annotations/emergency_residents.csv').values
y_student = pd.read_csv('./data/annotations/medical_students.csv').values

# Get true values
y_true = pd.read_csv('./data/annotations/gold_standard.csv').values

# get y_score for different models
y_score_list = [np.load('./dnn_predicts/other_seeds/model_' + str(i + 1) + '.npy') for i in range(10)]

# %% Get average model
# Get micro average precision
micro_avg_precision = [average_precision_score(y_true[:, :6], y_score[:, :6], average='micro')
                       for y_score in y_score_list]

# get ordered index
index = np.argsort(micro_avg_precision)
print('Micro average precision')
print(np.array(micro_avg_precision)[index])

# get 6th best model (immediately above median) out 10 different models
k_dnn_best = index[5]
y_score_best = y_score_list[k_dnn_best]

# Get threshold that yield the best precision recall using "get_optimal_precision_recall" on validation set
# (we rounded it up to three decimal cases to make it easier to read...)
threshold = np.array([0.124, 0.07, 0.05, 0.278, 0.390, 0.174])
mask = y_score_best > threshold

# Get neural network prediction
# This data was also saved in './data/annotations/dnn.csv'
y_neuralnet = np.zeros_like(y_score_best)
y_neuralnet[mask] = 1
y_neuralnet[mask] = 1

# %% Generate table with scores for the average model (Table 2)
scores_list = generate_table(y_true=y_true, score_fun=score_fun, diagnosis=diagnosis, y_neuralnet=y_neuralnet,
                             y_cardio=y_cardio, y_emerg=y_emerg, y_student=y_student)

# %% Plot precision recall curves (Figure 2)
plot_pre_rec_curve(y_true=y_true, k_dnn_best=k_dnn_best, diagnosis=diagnosis,
                   y_score_list=y_score_list, scores_list=scores_list,
                   predictor_names=predictor_names)

# %% Confusion matrices (Supplementary Table 1)
plot_confusion_matrix(y_true=y_true, nclasses=nclasses, diagnosis=diagnosis, y_neuralnet=y_neuralnet, y_cardio=y_cardio,
                      y_emerg=y_emerg, y_student=y_student)

# %% Compute scores and bootstraped version of these scores
bootstrap_nsamples = 1000
percentiles = [2.5, 97.5]

scores_percentiles_list, scores_resampled_list = compute_score_bootstraped(y_true=y_true,
                                                                           nclasses=nclasses,
                                                                           score_fun=score_fun,
                                                                           percentiles=percentiles,
                                                                           bootstrap_nsamples=bootstrap_nsamples,
                                                                           y_neuralnet=y_neuralnet,
                                                                           y_cardio=y_cardio,
                                                                           y_emerg=y_emerg,
                                                                           y_student=y_student,
                                                                           diagnosis=diagnosis,
                                                                           predictor_names=predictor_names)

# %% Print box plot (Supplementary Figure 1)
plot_box(scores_resampled_list=scores_resampled_list, predictor_names=predictor_names,
         bootstrap_nsamples=bootstrap_nsamples, score_fun=score_fun)

# %% McNemar test (Supplementary Table 3)
McNemar_score(y_true=y_true, y_neuralnet=y_neuralnet, y_cardio=y_cardio,
              y_emerg=y_emerg, y_student=y_student, diagnosis=diagnosis)

# %% Kappa score classifiers (Supplementary Table 2(a))
kappa_score_classifier(names=["DNN", "cardio.", "emerg.", "stud."],
                       predictors=[y_neuralnet, y_cardio, y_emerg, y_student],
                       diagnosis=diagnosis)

# %% Kappa score dataset generation (Supplementary Table 2(b))
kappa_score_dataset_generation(y_neuralnet=y_neuralnet, y_cardiologist1=y_cardiologist1,
                               y_cardiologist2=y_cardiologist2, diagnosis=diagnosis)

# %% Compute scores and bootstraped version of these scores on alternative splits
scores_resampled_list = compute_score_bootstraped_splits(y_true=y_true, y_score_best=y_score_best,
                                                         score_fun=score_fun, bootstrap_nsamples=bootstrap_nsamples,
                                                         percentiles=percentiles, diagnosis=diagnosis)

# %% Print box plot on alternative splits (Supplementary Figure 2 (a))
plot_box_splits(scores_resampled_list=scores_resampled_list,
                bootstrap_nsamples=bootstrap_nsamples,
                score_fun=score_fun)
