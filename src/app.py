import sys
import os
import re
import joblib
import tempfile
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QFileDialog,
    QLabel, QStackedWidget, QTextEdit, QMessageBox, QDialog
)
from PyQt5.QtCore import Qt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
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
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QFont

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Employee Performance Analysis")
        size = 64
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        font = QFont("Segoe UI Emoji", size // 2)
        painter.setFont(font)
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "📊")
        painter.end()

        self.setWindowIcon(QIcon(pixmap))

        self.setGeometry(300, 200, 1000, 700)

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.upload_widget = self.create_upload_widget()
        self.model_widget = self.create_model_widget()

        self.stacked_widget.addWidget(self.upload_widget)
        self.stacked_widget.addWidget(self.model_widget)

        self.file_path = ""
        self.data = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def create_upload_widget(self):
        """Widget for uploading XLS file"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignCenter)

        self.upload_icon = QLabel("🔍", self)
        self.upload_icon.setAlignment(Qt.AlignCenter)
        self.upload_icon.setFixedSize(500, 300)
        self.upload_icon.setStyleSheet("font-size: 200px;")
        self.label = QLabel("", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("padding: 20px 0;")
        self.button = QPushButton("Upload XLS File")
        self.button.setStyleSheet("font-size: 20px; padding: 20px 100px;")
        self.button.clicked.connect(self.upload_file)

        layout.addWidget(self.upload_icon)
        layout.addWidget(self.button)
        layout.addWidget(self.label)

        return widget

    def create_model_widget(self):
        """Widget with model buttons and result display"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignCenter)

        button_layout = QHBoxLayout()
        button_layout.setAlignment(Qt.AlignTop)

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
            "ANN": "ann",
            "Analysis Tool": "analysis"
        }

        for label, model_name in self.buttons.items():
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, m=model_name: self.run_model(m))
            btn.setFixedHeight(50)
            btn.setStyleSheet("""
                QPushButton {
                    color: #666666;
                    border-radius: 10px;
                    border: 2px solid #000000;
                    font: bold 14px;
                    padding: 10px 20px;
                }
            """)
            button_layout.addWidget(btn)

        self.result_display = QTextEdit(self)
        self.result_display.setAlignment(Qt.AlignCenter)
        self.result_display.setFixedSize(500, 500)
        self.result_display.setStyleSheet("font-size: 20px; margin-bottom: 20px;")
        self.result_display.setReadOnly(True)

        result_display_layout = QHBoxLayout()
        result_display_layout.addWidget(self.result_display)

        self.download_button = QPushButton("Download ANN Model")
        download_button_layout = QHBoxLayout()
        download_button_layout.setAlignment(Qt.AlignCenter)
        download_button_layout.addWidget(self.download_button)

        self.download_button.setFixedSize(200, 100)
        self.download_button.setStyleSheet("""
            QPushButton {
                color: #000000;
                border-radius: 10px;
                border: 2px solid #000000;
                font: bold 14px;
                margin-top: 10px;
            }
        """)
        self.download_button.setEnabled(False)
        self.download_button.clicked.connect(self.download_model)
        self.download_button.hide()

        layout.addLayout(button_layout)
        layout.addLayout(result_display_layout)
        layout.addLayout(download_button_layout)
        layout.setAlignment(Qt.AlignHCenter)
        # button_layout.setAlignment(Qt.AlignHCenter)

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
            self.stacked_widget.setCurrentIndex(1)
        else:
            QMessageBox.warning(self, "No File", "Please select an XLS file!")

    def preprocess_data(self):
        """Preprocess the uploaded XLS data"""
        try:
            self.data = pd.read_excel(self.file_path)
            self.perf_graph_data = self.data.iloc[:,[5,27]].copy()

            enc = LabelEncoder()
            for col in (2, 3, 4, 5, 6, 7, 16, 26):
                self.data.iloc[:, col] = enc.fit_transform(self.data.iloc[:, col])

            self.data.drop(['EmpNumber'], inplace=True, axis=1)

            y = self.data['PerformanceRating']
            X = self.data.iloc[:, [4, 5, 9, 16, 20, 21, 22, 23, 24]]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

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
        elif model_name == "analysis":
            self.analysis_tool()

        if self.X_train is None or self.y_train is None:
            QMessageBox.warning(self, "No Data", "Please upload and preprocess the data first.")
            return

        try:
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

            if model_name == "xgb":
                le = LabelEncoder()
                y_train_encoded = le.fit_transform(self.y_train)
                y_test_encoded = le.transform(self.y_test)
            else:
                y_train_encoded = self.y_train
                y_test_encoded = self.y_test

            model.fit(self.X_train, y_train_encoded)
            y_pred = model.predict(self.X_test)

            if model_name == "ann":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".ml") as tmp:
                    joblib.dump(model, tmp.name)
                    self.model_path = tmp.name
                self.download_button.show()
                self.download_button.setEnabled(True)
            else:
                self.download_button.hide()

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
        dialog.setGeometry(200, 200, 1100, 600)

        layout = QVBoxLayout(dialog)

        dept = self.perf_graph_data
        dept_per = dept.copy()

        sns.set_theme(style="whitegrid")
        colors = sns.color_palette("Set2")

        fig, ax = plt.subplots(figsize=(10, 4.5))
        sns.barplot(data=dept_per, x='EmpDepartment', y='PerformanceRating', palette=colors, ax=ax)
        ax.set_title('Performance Rating by Department')
        ax.set_xlabel('Department')
        ax.set_ylabel('Performance Rating')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        fig.tight_layout()

        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        dialog.exec_()

    def show_correlation_matrix(self):
        """Display feature correlation matrix"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Correlation Matrix")
        dialog.setGeometry(200, 200, 1400, 800)

        layout = QVBoxLayout(dialog)
        sns.set(font_scale=0.8)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(self.data.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)

        fig.tight_layout()

        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        dialog.exec_()

    def analysis_tool(self):
        """Open analysis tool dialog"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Analysis Tool")
        dialog.setGeometry(200, 200, 800, 600)

        layout = QVBoxLayout(dialog)
        layout.setAlignment(Qt.AlignTop)

        self.analysis_input = QTextEdit(self)
        self.analysis_input.setPlaceholderText("Enter Employee Number (e.g., E1234567)")
        self.analysis_input.setFixedSize(400, 50)

        hhlayout = QHBoxLayout()
        hhlayout.addWidget(self.analysis_input)
        hhlayout.setAlignment(Qt.AlignCenter)

        self.analysis_button = QPushButton("Analyze")
        self.analysis_button.setStyleSheet("font-size: 20px;")
        self.analysis_button.setFixedWidth(200)
        self.analysis_button.clicked.connect(self.analysis)

        hlayout = QHBoxLayout()
        hlayout.addWidget(self.analysis_button)
        hlayout.setAlignment(Qt.AlignCenter)

        self.analysis_text = QTextEdit(self)
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setStyleSheet("font-size: 16px; padding: 10px;")
        self.analysis_text.setText("Analysis Tool is under construction.")
        self.analysis_text.setFixedHeight(500)

        layout.addLayout(hhlayout)
        layout.addLayout(hlayout)
        layout.addWidget(self.analysis_text)

        dialog.exec_()

    def analysis(self):
        """Perform analysis on the entered employee number"""
        emp_number = self.analysis_input.toPlainText().strip()
        if not emp_number or not self.validate_emp_number(emp_number):
            self.analysis_text.setText("Invalid Employee Number format. Please enter a valid number (e.g., E1234567).")
            return
        
        database = pd.read_excel(self.file_path)

        for column in ['Gender', 'EducationBackground', 'MaritalStatus', 'EmpDepartment', 
               'EmpJobRole', 'BusinessTravelFrequency', 'OverTime', 'Attrition']:
            le = LabelEncoder()
            database[column] = le.fit_transform(database[column])

        X = database[['Age', 'EmpJobLevel', 'YearsSinceLastPromotion', 
                'EmpEnvironmentSatisfaction', 'EmpJobInvolvement', 
                'EmpJobSatisfaction', 'EmpWorkLifeBalance', 
                'ExperienceYearsAtThisCompany', 'ExperienceYearsInCurrentRole', 
                'YearsWithCurrManager']]
        y = database['PerformanceRating']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        param_grid = {
            'criterion': ['gini', 'entropy'],
            'min_samples_split': [2, 3, 4, 5],
            'min_samples_leaf': [1, 2, 3]
        }

        grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42),
                           param_grid,
                           scoring='accuracy',
                           cv=3,
                           verbose=1)
        grid_search.fit(X_train, y_train)
        dt_model = grid_search.best_estimator_

        def predict_and_display(emp_number, model):
            """Predict and display results for an employee"""
            emp_record = database[database['EmpNumber'] == emp_number]
            
            if emp_record.empty:
                print(f"\n🚫 Employee {emp_number} not found in records")
                return

            # Prepare features and predict
            features = emp_record[X.columns]
            prediction = model.predict(features)[0]
            
            output = []
            output.append("\n" + "="*50)
            output.append(f"📊 Performance Prediction for Employee {emp_number}")
            output.append("="*50)
            
            output.append("\n🔍 Employee Details:")
            output.append(f"• Department:       {emp_record['EmpDepartment'].values[0]}")
            output.append(f"• Job Role:         {emp_record['EmpJobRole'].values[0]}")
            output.append(f"• Age:              {emp_record['Age'].values[0]}")
            output.append(f"• Years in Company: {emp_record['ExperienceYearsAtThisCompany'].values[0]}")
            
            output.append("\n📈 Performance Prediction:")
            output.append(f"• Predicted Rating: {prediction}/5")
            output.append(f"• Promotion Recommendation: {'✅ Approved' if prediction >= 4 else '❌ Not Approved'}")
            
            output.append("\n📋 Key Performance Indicators:")
            output.append(f"• Job Involvement:     {emp_record['EmpJobInvolvement'].values[0]}/4")
            output.append(f"• Job Satisfaction:    {emp_record['EmpJobSatisfaction'].values[0]}/4")
            output.append(f"• Work-Life Balance:   {emp_record['EmpWorkLifeBalance'].values[0]}/4")
            output.append(f"• Last Promotion:      {emp_record['YearsSinceLastPromotion'].values[0]} years ago")
            
            output.append("\n" + "="*50 + "\n")
            
            return "\n".join(output)
        
        result = predict_and_display(emp_number, dt_model)

        self.analysis_text.setText(result)
        self.analysis_text.setAlignment(Qt.AlignLeft)
        self.analysis_text.setStyleSheet("font-size: 16px; padding: 10px;")
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setTextInteractionFlags(Qt.TextBrowserInteraction | Qt.TextSelectableByMouse)

    def validate_emp_number(self, emp_number):
        """Validate employee number format (E followed by 7 digits)"""
        return re.match(r'^E\d{7}$', emp_number) is not None
        
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
