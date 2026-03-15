import json
import os
import sys
import time

import numpy as np
import pandas as pd

# Path
input_dir = '/app/input_data/'
output_dir = '/app/output/'
sys.path.append('/app/ingested_program')

if __name__ == '__main__':
    from submission import Model

    start = time.time()

    # Read data
    print('Reading data')
    # Read data
    train = pd.read_csv(os.path.join(input_dir, 'train.csv'))
    X_train = train.drop(['label'], axis=1)
    y_train = train.filter(['label'], axis=1)
    X_test = pd.read_csv(os.path.join(input_dir, 'test.csv'))
    # Convert to numpy arrays
    X_train, y_train, X_test = np.array(X_train), np.array(y_train), np.array(X_test)
    # Initialize model
    print('Initializing the model')
    m = Model()
    # Train model
    print('Training the model')
    m.fit(X_train, y_train)
    # Make predictions
    print('Making predictions')
    y_pred = m.predict(X_test)
    # Save predictions
    np.savetxt(os.path.join(output_dir, 'data.predict'), y_pred)
    duration = time.time() - start
    print(f'Time elapsed so far: {duration}')
    # End
    duration = time.time() - start
    print(f'Completed. Total duration: {duration}')
    with open(os.path.join(output_dir, 'metadata.json'), 'w+') as f:
        json.dump({'duration': duration}, f)
    print('Ingestion program finished. Moving on to scoring')
    print()