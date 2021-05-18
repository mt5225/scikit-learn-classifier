from sklearn import datasets, svm, metrics
import time
import statistics as st
import pprint as pp

def create_model():
    
    return svm.SVC(gamma=0.001) # UPDATE

def benchmark_model(model, repeats=10):
    
    digits = datasets.load_digits()

    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1)) # MNIST images

    expected = digits.target[n_samples // 2:]

    output = {"Training time (s)": [], "Prediction time (s)": [],
    "Performance (micro avg f1 score)": []}
    
    tt, pt, p = [], [], []
    
    for i in range(0, repeats):
        
        # Train the classifier model
        start = time.time()
        model.fit(data[:n_samples // 2], digits.target[:n_samples // 2])
        end = time.time()
        tt.append(end - start)
        
        # Use the model for prediction
        start = time.time()
        predicted = model.predict(data[n_samples // 2:])
        end = time.time()
        pt.append(end - start)
        p.append(metrics.classification_report(expected, predicted, output_dict=True)['macro avg']['f1-score'])
        
    # Get median benchmarks for chosen number of repeats
    
    benchmarks = {
        "Training time (s)": st.median(tt),
        "Prediction time (s)": st.median(pt),
        "Performance (micro avg f1 score)": st.median(p)
    }
    
    return benchmarks
