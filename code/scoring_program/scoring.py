# Imports
import json
import os
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# Path
reference_dir = os.path.join('/app/input/', 'ref')
prediction_dir = os.path.join('/app/input/', 'res')
score_file = os.path.join('/app/output/', 'scores.json')
html_file = os.path.join('/app/output/', 'detailed_results.html')

if __name__ == '__main__':
    # Initialized detailed results
    with open(html_file, 'a', encoding="utf-8") as fh:
        fh.write('<h1>Detailed results</h1>')
    
    print()
    print('Reading prediction')
    y_test = pd.read_csv(os.path.join(reference_dir, 'test_labels.csv'))
    y_test = np.array(y_test)
    y_pred = np.genfromtxt(os.path.join(prediction_dir, 'data.predict'))
    # Compute score
    accuracy = accuracy_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred)
    print('Accuracy: {}'.format(accuracy))
    print('F1-Score: {}'.format(f1score))

    # Get duration
    with open(os.path.join(prediction_dir, 'metadata.json')) as f:
        duration = json.load(f).get('duration', -1)
    print('Duration: {}'.format(duration))

    # Write scores
    print()
    print('Scoring program finished. Writing scores.')
    with open(score_file, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps({'accuracy':accuracy, 'f1score':f1score, 'duration':duration}))
