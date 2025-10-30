import os
import sys
import logging
import json
import numpy as np
from flask import Flask, render_template, request, Response

from .preprocessing import extract_fbanks
from .predictions import get_embeddings, get_cosine_distance
from .adaptive_threshold import AdaptiveThresholdSystem, calculate_confidence_score

# Initialize adaptive threshold system
adaptive_threshold_system = AdaptiveThresholdSystem(
    base_threshold=0.45,
    min_threshold=0.30,
    max_threshold=0.60
)

app = Flask(__name__)

DATA_DIR = 'data_files/'
sys.path.append('..')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/login/<string:username>', methods=['POST'])
def login(username):
    filename = _save_file(request, username)
    fbanks = extract_fbanks(filename)
    embeddings = get_embeddings(fbanks)
    
    # Load stored embeddings
    embeddings_file = DATA_DIR + username + '/embeddings.npy'
    stored_embeddings = np.load(embeddings_file)
    stored_embeddings = stored_embeddings.reshape((1, -1))

    # Calculate distances
    distances = get_cosine_distance(embeddings, stored_embeddings)
    mean_distance = np.mean(distances)
    
    # IMPROVEMENT 3: Use adaptive threshold instead of fixed
    # Fixed: use get_adaptive_threshold, not get_threshold
    user_threshold, quality = adaptive_threshold_system.get_adaptive_threshold(
        username,
        enrollment_features=None  # Will load from stored data
    )
    
    # IMPROVEMENT 4: Calculate confidence score
    confidence = calculate_confidence_score(distances, user_threshold)
    
    print(f'Mean distance: {mean_distance:.4f}', flush=True)
    print(f'Threshold: {user_threshold:.4f}', flush=True)
    print(f'Confidence: {confidence:.2%}', flush=True)
    
    # Make decision
    positives = distances < user_threshold
    positives_mean = np.mean(positives)
    print('Positives ratio: {:.2%}'.format(positives_mean), flush=True)
    
    # Return result with confidence
    if positives_mean >= .65:
        result = {
            'status': 'SUCCESS',
            'confidence': float(confidence),
            'mean_distance': float(mean_distance),
            'threshold': float(user_threshold)
        }
        return Response(json.dumps(result), mimetype='application/json')
    else:
        result = {
            'status': 'FAILURE',
            'confidence': float(confidence),
            'mean_distance': float(mean_distance),
            'threshold': float(user_threshold)
        }
        return Response(json.dumps(result), mimetype='application/json')


@app.route('/register/<string:username>', methods=['POST'])
def register(username):
    filename = _save_file(request, username)
    fbanks = extract_fbanks(filename)
    embeddings = get_embeddings(fbanks)
    print('shape of embeddings: {}'.format(embeddings.shape), flush=True)
    
    # IMPROVEMENT 1: Store individual samples for quality assessment
    dir_ = DATA_DIR + username
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    np.save(dir_ + '/enrollment_samples.npy', embeddings)
    
    # Calculate and store mean embeddings (original behavior)
    mean_embeddings = np.mean(embeddings, axis=0)
    np.save(dir_ + '/embeddings.npy', mean_embeddings)
    
    # IMPROVEMENT 2: Calculate adaptive threshold and quality
    user_threshold, quality = adaptive_threshold_system.get_adaptive_threshold(
        username,
        enrollment_features=embeddings,
        save=True
    )
    
    # Get quality feedback
    quality_feedback = adaptive_threshold_system.get_quality_feedback(quality)
    
    quality_message = f"Enrollment quality: {quality:.2%} - {quality_feedback}"
    quality_message += f" | Your personalized threshold: {user_threshold:.3f}"
    
    print(quality_message, flush=True)
    
    # Save thresholds to disk
    adaptive_threshold_system.save_thresholds()
    
    return Response(quality_message, mimetype='application/json')


def _save_file(request_, username):
    file = request_.files['file']
    dir_ = DATA_DIR + username
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    filename = DATA_DIR + username + '/sample.wav'
    file.save(filename)
    return filename


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)