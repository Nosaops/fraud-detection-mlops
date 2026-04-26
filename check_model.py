import pickle
import numpy as np

print("Loading model...")
with open("gs_rf.pkl", "rb") as f:
    model = pickle.load(f)

print(f"Model type: {type(model)}")
print(f"Model loaded successfully")

# Test with one fake row of 30 features (Time + V1-V28 + Amount)
test_row = np.array([[0.0, -1.35, -0.07, 2.53, 1.37, -0.33,
                      0.46, 0.23, 0.09, 0.36, 0.09, -0.55,
                      -0.61, -0.99, -0.31, 1.46, -0.47, 0.20,
                      0.02, 0.40, 0.25, -0.01, 0.27, -0.11,
                      0.06, -0.18, -0.14, -0.06, -0.06, 149.62]])

print(f"Test input shape: {test_row.shape}")
prediction = model.predict(test_row)
probability = model.predict_proba(test_row)

print(f"Prediction: {prediction[0]} ({'FRAUD' if prediction[0] == 1 else 'LEGITIMATE'})")
print(f"Fraud probability: {probability[0][1]:.4f}")
print("Model check complete!")
