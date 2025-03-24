import sys
import os
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QFileDialog,
    QLabel, QStackedWidget, QTextEdit, QMessageBox, QDialog
)
from PyQt5.QtCore import Qt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Employee Performance Analysis")
        self.setGeometry(300, 200, 1000, 700)

        # Stacked widget for switching between upload and model widgets
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # Upload and model widgets
        self.upload_widget = self.create_upload_widget()
        self.model_widget = self.create_model_widget()

        # Add widgets to the stack
        self.stacked_widget.addWidget(self.upload_widget)
        self.stacked_widget.addWidget(self.model_widget)

        # Variables
        self.file_path = ""
        self.data = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def create_upload_widget(self):
        """Widget for uploading XLS file"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.label = QLabel("", self)
        self.button = QPushButton("Upload XLS File")
        self.button.clicked.connect(self.upload_file)

        layout.addWidget(self.button)
        layout.addWidget(self.label)

        return widget

    def create_model_widget(self):
        """Widget with model buttons and result display"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Buttons for models
        button_layout = QHBoxLayout()

        self.buttons = {
            "Graph: Department-Wise": "dept_graph",
            "Graph: Correlation Matrix": "corr_graph",
            "Logistic Regression": "logistic",
            "SVM": "svm",
            "Decision Tree": "dtree",
            "Random Forest": "rf",
            "Naive Bayes": "nb",
            "KNN": "knn",
            "XGBoost": "xgb",
            "ANN": "ann"
        }

        for label, model_name in self.buttons.items():
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, m=model_name: self.run_model(m))
            button_layout.addWidget(btn)

        # Result display
        self.result_display = QTextEdit(self)
        self.result_display.setReadOnly(True)

        # Download button (initially hidden)
        self.download_button = QPushButton("Download ANN Model")
        self.download_button.setEnabled(False)
        self.download_button.clicked.connect(self.download_model)
        self.download_button.hide()  # Hide initially

        # Add the result display first
        layout.addWidget(self.result_display)

        # Store the download button separately to add it later
        self.download_button = QPushButton("Download ANN Model")
        self.download_button.setEnabled(False)
        self.download_button.clicked.connect(self.download_model)
        self.download_button.hide()  # Initially hidden

        # Layout arrangement
        layout.addLayout(button_layout)
        layout.addWidget(self.result_display)
        # Finally, add the download button at the bottom
        layout.addWidget(self.download_button)

        return widget

    def upload_file(self):
        """Upload XLS file and preprocess it"""
        options = QFileDialog.Options()
        file_filter = "Excel Files (*.xls *.xlsx)"
        file_name, _ = QFileDialog.getOpenFileName(self, "Select XLS File", "", file_filter, options=options)

        if file_name:
            self.file_path = file_name
            self.label.setText(f"File uploaded: {file_name}")
            self.preprocess_data()
            self.stacked_widget.setCurrentIndex(1)  # Switch to model widget
        else:
            QMessageBox.warning(self, "No File", "Please select an XLS file!")

    def preprocess_data(self):
        """Preprocess the uploaded XLS data"""
        try:
            # Load data
            self.data = pd.read_excel(self.file_path)

            # Encode categorical columns
            enc = LabelEncoder()
            for col in (2, 3, 4, 5, 6, 7, 16, 26):
                self.data.iloc[:, col] = enc.fit_transform(self.data.iloc[:, col])

            # Drop unnecessary columns
            self.data.drop(['EmpNumber'], inplace=True, axis=1)

            # Select features and target
            y = self.data['PerformanceRating']
            X = self.data.iloc[:, [4, 5, 9, 16, 20, 21, 22, 23, 24]]

            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

            # Standardization
            sc = StandardScaler()
            self.X_train = sc.fit_transform(X_train)
            self.X_test = sc.transform(X_test)
            self.y_train = y_train
            self.y_test = y_test

            QMessageBox.information(self, "Success", "Data preprocessed successfully!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to preprocess file: {e}")

    def run_model(self, model_name):
        """Run the selected model or plot graphs"""
        if model_name == "dept_graph":
            self.show_department_graph()
            return
        elif model_name == "corr_graph":
            self.show_correlation_matrix()
            return

        if self.X_train is None or self.y_train is None:
            QMessageBox.warning(self, "No Data", "Please upload and preprocess the data first.")
            return

        try:
            # Select the appropriate model
            models = {
                "logistic": LogisticRegression(),
                "svm": SVC(kernel='rbf', C=100, random_state=10),
                "dtree": DecisionTreeClassifier(random_state=42),
                "rf": RandomForestClassifier(random_state=33, n_estimators=23),
                "nb": BernoulliNB(),
                "knn": KNeighborsClassifier(n_neighbors=10, metric='euclidean'),
                "xgb": XGBClassifier(),
                "ann": MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=2000, random_state=10)
            }

            model = models[model_name]

            # Normalize labels for XGBoost
            if model_name == "xgb":
                le = LabelEncoder()
                y_train_encoded = le.fit_transform(self.y_train)
                y_test_encoded = le.transform(self.y_test)
            else:
                y_train_encoded = self.y_train
                y_test_encoded = self.y_test

            # Train and predict
            model.fit(self.X_train, y_train_encoded)
            y_pred = model.predict(self.X_test)

            if model_name == "ann":
                self.model_path = "INX_Future_Inc_ANN.ml"
                joblib.dump(model, self.model_path)
                self.download_button.show()  # Show the download button
                self.download_button.setEnabled(True)
            else:
                self.download_button.hide()  # Hide the button for other models

            accuracy = accuracy_score(y_test_encoded, y_pred)
            report = classification_report(y_test_encoded, y_pred, zero_division=0)
            matrix = confusion_matrix(y_test_encoded, y_pred)

            result = (
                f"Model: {model_name.upper()}\n"
                f"Accuracy: {accuracy:.4f}\n\n"
                f"{report}\n"
                f"Confusion Matrix:\n{matrix}\n"
            )
            self.result_display.setText(result)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run model: {e}")

    def show_department_graph(self):
        """Display department-wise performance graph"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Department-Wise Performance")
        dialog.setGeometry(200, 200, 800, 600)

        layout = QVBoxLayout(dialog)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='EmpDepartment', y='PerformanceRating', data=self.data, ax=ax)

        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        dialog.exec_()

    def show_correlation_matrix(self):
        """Display feature correlation matrix"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Correlation Matrix")
        dialog.setGeometry(200, 200, 800, 600)

        layout = QVBoxLayout(dialog)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(self.data.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)

        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        dialog.exec_()

    def download_model(self):
        """Download the trained ANN model"""
        if hasattr(self, 'model_path') and os.path.exists(self.model_path):
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Model", "", "Model Files (*.ml)")
            
            if save_path:
                joblib.dump(joblib.load(self.model_path), save_path)
                QMessageBox.information(self, "Download Complete", f"Model saved at: {save_path}")
            else:
                QMessageBox.warning(self, "Cancelled", "Download cancelled.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
