import numpy as np
from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def train(X_train, y_train, X_val, y_val):
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    
    ensemble = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(C=0.5, max_iter=1000, random_state=42)),
            ("svc", SVC(C=1.5, kernel="rbf", gamma="scale", probability=True, random_state=42)),
            ("et", ExtraTreesClassifier(n_estimators=100, max_depth=6, random_state=42))
        ],
        voting="soft",
        weights=[1, 2, 1]  # Weight the RBF SVM slightly higher
    )
    
    ensemble.fit(X_full, y_full)
    return ensemble.predict