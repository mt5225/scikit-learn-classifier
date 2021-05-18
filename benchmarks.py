import classifier
model = classifier.create_model()
print(classifier.benchmark_model(model, repeats=10))
