import pandas as pd
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score # pyright: ignore[reportMissingModuleSource]
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# ------------------------------ 
# LOAD + PREPROCESS FUNCTION
# ------------------------------
def load_and_preprocess(file_path):

    

    # Load CSV
    df = pd.read_csv(file_path)

    # Features and label
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


# ------------------------------ 
# MAIN CODE
# ------------------------------
if __name__ == "__main__":

    # Correct filepath
    X_train, X_test, y_train, y_test = load_and_preprocess("data/creditcard.csv")

    # ---------------- Logistic Regression ----------------
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)
    log_pred = log_model.predict(X_test)

    print("\n==== Logistic Regression ====")
    print(classification_report(y_test, log_pred))
    print("AUC:", roc_auc_score(y_test, log_pred))


    # ---------------- KNN ----------------
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)

    print("\n==== KNN ====")
    print(classification_report(y_test, knn_pred))
    print("AUC:", roc_auc_score(y_test, knn_pred))


    # ---------------- SVM ----------------
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)

    print("\n==== SVM ====")
    print(classification_report(y_test, svm_pred))
    print("AUC:", roc_auc_score(y_test, svm_pred))


    # ---------------- Decision Tree ----------------
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)
    tree_pred = tree.predict(X_test)

    print("\n==== Decision Tree ====")
    print(classification_report(y_test, tree_pred))
    print("AUC:", roc_auc_score(y_test, tree_pred))
