import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Simulate EAR-like data for 3 classes
np.random.seed(42)
n_samples = 300

# EAR ranges (hypothetical)
active = np.random.normal(loc=0.32, scale=0.02, size=n_samples)
drowsy = np.random.normal(loc=0.26, scale=0.015, size=n_samples)
sleepy = np.random.normal(loc=0.20, scale=0.01, size=n_samples)

# Combine into dataset
X = np.concatenate([active, drowsy, sleepy])
y_true = np.array([0]*n_samples + [1]*n_samples + [2]*n_samples)  # 0: Active, 1: Drowsy, 2: Sleepy

# Combined EAR Classifier (simple threshold logic)
y_pred_ear = []
for val in X:
    if val > 0.29:
        y_pred_ear.append(0)  # Active
    elif val > 0.23:
        y_pred_ear.append(1)  # Drowsy
    else:
        y_pred_ear.append(2)  # Sleepy
y_pred_ear = np.array(y_pred_ear)

# Accuracy
acc_ear = accuracy_score(y_true, y_pred_ear)
print(f"Accuracy : {acc_ear * 100:.2f}%\n")

# Classification Report (Extract precision, recall, f1 only)
report = classification_report(
    y_true, y_pred_ear, target_names=['Active', 'Drowsy', 'Sleepy'], output_dict=True, zero_division=0
)

# Remove 'support', extract precision, recall, f1-score
labels = ['Active', 'Drowsy', 'Sleepy']
precision = [report[label]['precision'] for label in labels]
recall = [report[label]['recall'] for label in labels]
f1 = [report[label]['f1-score'] for label in labels]

# Print metrics (without support)
print("Precision, Recall, F1-score (per class):\n")
for i in range(len(labels)):
    print(f"{labels[i]} - Precision: {precision[i]:.2f}, Recall: {recall[i]:.2f}, F1-score: {f1[i]:.2f}")

# --- Bar Graphs ---
x = np.arange(len(labels))
width = 0.25

plt.figure(figsize=(10, 6))
plt.bar(x - width, precision, width, label='Precision', color='skyblue')
plt.bar(x, recall, width, label='Recall', color='orange')
plt.bar(x + width, f1, width, label='F1-score', color='limegreen')

plt.xticks(x, labels)
plt.ylim(0, 1.1)
plt.ylabel("Score")
plt.title("Combined EAR Model - Performance Metrics")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Histogram of EAR values
plt.figure(figsize=(10, 6))
plt.hist(active, bins=20, alpha=0.6, label='Active', color='green')
plt.hist(drowsy, bins=20, alpha=0.6, label='Drowsy', color='orange')
plt.hist(sleepy, bins=20, alpha=0.6, label='Sleepy', color='red')
plt.title('Simulated EAR Distributions')
plt.xlabel('EAR Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred_ear)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.title('Combined EAR Model - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()
