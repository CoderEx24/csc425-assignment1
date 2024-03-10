from sklearn.metrics import accuracy_score, classification_report
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import sklearn

data = pd.read_csv('./diabetes.csv')

inputs = data.drop('Outcome', axis=1)
outputs = data['Outcome']

standard_inputs = StandardScaler().fit_transform(inputs)

reduced_inputs = PCA(n_components=5).fit_transform(standard_inputs)

train_x, test_x, train_y, test_y = train_test_split(reduced_inputs, outputs, test_size=0.25)

model_classes = [
        ("Random Forests", sklearn.ensemble.RandomForestClassifier),
        ("Naive Bayes", sklearn.naive_bayes.GaussianNB),
        ("Gaussian Process", sklearn.gaussian_process.GaussianProcessClassifier),
]

max_accuracy = (-1, '')

for model, klass in model_classes:
    model = klass()
    model.fit(train_x, train_y)
    pred = model.predict(test_x)

    accuracy = accuracy_score(test_y, pred)
    max_accuracy = (max(max_accuracy[0], accuracy), model)

    print('=' * 10)
    print(f'Accuracy Score: {accuracy}')
    print(f'Classification Report:\n{classification_report(test_y, pred)}')
    print('=' * 10)

print(f'{max_accuracy = }')


