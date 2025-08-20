import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog,
                             QTabWidget, QTableWidget, QTableWidgetItem, QLabel, QHBoxLayout, QComboBox, QLineEdit)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from scipy.special import erfc


class EPMAViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EPMA/EDX Diffusion Profile Analyzer")
        self.resize(1000, 750)

        self.tabs = QTabWidget()
        self.data_tab = QWidget()
        self.heatmap_tab = QWidget()
        self.vector_tab = QWidget()
        self.surface3d_tab = QWidget()

        self.tabs.addTab(self.data_tab, "Data Table")
        self.tabs.addTab(self.heatmap_tab, "Heatmap View")
        self.tabs.addTab(self.vector_tab, "Vector Field View")
        self.tabs.addTab(self.surface3d_tab, "3D Surface Plot")

        self.data = None
        self.gradient_data = None
        self.vector_toggle = True
        self.heatmap_colorbar = None

        self.init_data_tab()
        self.init_heatmap_tab()
        self.init_vector_tab()
        self.init_surface3d_tab()

        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def init_data_tab(self):
        layout = QVBoxLayout()

        self.load_button = QPushButton("Browse and Load CSV File")
        self.load_button.clicked.connect(self.load_csv)
        layout.addWidget(self.load_button)

        self.table_widget = QTableWidget()
        layout.addWidget(self.table_widget)

        self.data_tab.setLayout(layout)

    def init_heatmap_tab(self):
        layout = QVBoxLayout()

        self.plot_button = QPushButton("Show Heatmap ")
        self.plot_button.clicked.connect(self.plot_heatmap)
        layout.addWidget(self.plot_button)

        self.result_label = QLabel("")
        layout.addWidget(self.result_label)

        self.canvas = FigureCanvas(Figure(figsize=(6, 4)))
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.ax = self.canvas.figure.add_subplot(111)

        self.heatmap_tab.setLayout(layout)

    def init_vector_tab(self):
        layout = QVBoxLayout()

        self.vector_plot_button = QPushButton("Show Vector Field")
        self.vector_plot_button.clicked.connect(self.plot_vector_field)
        layout.addWidget(self.vector_plot_button)

        self.k_input = QComboBox()
        self.k_input.addItems([str(i) for i in range(2, 7)])  # Allow user to pick k = 2 to 6
        layout.addWidget(QLabel("Select number of clusters (k):"))
        layout.addWidget(self.k_input)

        self.model_selector = QComboBox()
        self.model_selector.addItems(["Auto Suggest", "1D erfc", "2D Gaussian", "Exponential Decay", "2D Radial", "1D Slab", "Cylindrical", "Spherical"])
        layout.addWidget(QLabel("Select Diffusion Model:"))
        layout.addWidget(self.model_selector)

        self.time_input = QLineEdit()
        self.time_input.setPlaceholderText("Enter time (t)")
        layout.addWidget(QLabel("Diffusion Time (t):"))
        layout.addWidget(self.time_input)

        self.analyze_button = QPushButton("Compute ∇C and Dominant Angles")
        self.analyze_button.clicked.connect(self.analyze_gradient_angles)
        layout.addWidget(self.analyze_button)

        self.elbow_button = QPushButton("Determine Optimal k via Elbow Method")
        self.elbow_button.clicked.connect(self.find_optimal_k)
        layout.addWidget(self.elbow_button)

        self.export_button = QPushButton("Export ∇C Data as CSV")
        self.export_button.clicked.connect(self.export_gradient_csv)
        layout.addWidget(self.export_button)

        self.fit_button = QPushButton("Fit Diffusion Model Along Dominant Direction")
        self.fit_button.clicked.connect(self.fit_selected_model)
        layout.addWidget(self.fit_button)

        self.result_label = QLabel("")
        layout.addWidget(self.result_label)

        self.vector_canvas = FigureCanvas(Figure(figsize=(6, 6)))
        self.vector_toolbar = NavigationToolbar2QT(self.vector_canvas, self)
        layout.addWidget(self.vector_toolbar)
        layout.addWidget(self.vector_canvas)
        self.vector_ax = self.vector_canvas.figure.add_subplot(111)

        self.vector_tab.setLayout(layout)

    def get_time_value(self):
        try:
            t_val = float(self.time_input.text())
            if t_val <= 0:
                raise ValueError
            return t_val
        except:
            self.result_label.setText("⚠️ Invalid time value. Using default t = 1.")
            return 1.0

    def init_surface3d_tab(self):
        layout = QVBoxLayout()

        self.plot3d_button = QPushButton("Show 3D Surface Plot ")
        self.plot3d_button.clicked.connect(self.plot_surface3d)
        layout.addWidget(self.plot3d_button)

        view_buttons_layout = QHBoxLayout()
        self.view_top_button = QPushButton("Top View")
        self.view_top_button.clicked.connect(lambda: self.set_3d_view(90, -90))
        view_buttons_layout.addWidget(self.view_top_button)

        self.view_side_button = QPushButton("Side View")
        self.view_side_button.clicked.connect(lambda: self.set_3d_view(0, 0))
        view_buttons_layout.addWidget(self.view_side_button)

        self.view_iso_button = QPushButton("Isometric View")
        self.view_iso_button.clicked.connect(lambda: self.set_3d_view(30, 45))
        view_buttons_layout.addWidget(self.view_iso_button)

        self.rotate_button = QPushButton("Start Rotation")
        self.rotate_button.setCheckable(True)
        self.rotate_button.toggled.connect(self.toggle_rotation)
        view_buttons_layout.addWidget(self.rotate_button)

        layout.addLayout(view_buttons_layout)

        self.canvas3d = FigureCanvas(Figure(figsize=(6, 4)))
        self.canvas3d.setFocusPolicy(Qt.ClickFocus)
        self.canvas3d.setFocus()
        self.toolbar3d = NavigationToolbar2QT(self.canvas3d, self)
        layout.addWidget(self.toolbar3d)
        layout.addWidget(self.canvas3d)
        self.ax3d = self.canvas3d.figure.add_subplot(111, projection='3d')

        self.surface3d_tab.setLayout(layout)

    def load_csv(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)", options=options)
        if file_name:
            self.data = pd.read_csv(file_name)
            self.display_data()

    def display_data(self):
        if self.data is not None:
            self.table_widget.setRowCount(len(self.data))
            self.table_widget.setColumnCount(len(self.data.columns))
            self.table_widget.setHorizontalHeaderLabels(self.data.columns)

            for i in range(len(self.data)):
                for j in range(len(self.data.columns)):
                    item = QTableWidgetItem(str(self.data.iat[i, j]))
                    item.setTextAlignment(Qt.AlignCenter)
                    self.table_widget.setItem(i, j, item)

    def compute_gradient_field(self):
        if self.data is not None:
            x = self.data.iloc[:, 0].values
            y = self.data.iloc[:, 1].values
            c = self.data.iloc[:, 2].values

            df_pivot = pd.DataFrame({'x': x, 'y': y, 'c': c})
            pivot = df_pivot.pivot_table(index='y', columns='x', values='c')
            pivot = pivot.fillna(0)  # Fill missing values with 0

            try:
                grad_y, grad_x = np.gradient(pivot.values)
            except Exception as e:
                print("Gradient computation failed:", e)
                return None, None, None, None, None

            X, Y = np.meshgrid(pivot.columns.values, pivot.index.values)

            self.gradient_data = pd.DataFrame({
                'X': X.flatten(), 'Y': Y.flatten(),
                'dC/dx': grad_x.flatten(), 'dC/dy': grad_y.flatten(),
                'Angle(deg)': np.degrees(np.arctan2(grad_y.flatten(), grad_x.flatten()))
            })
            return X, Y, pivot.values, grad_x, grad_y
        return None, None, None, None, None

    def plot_heatmap(self):
        X, Y, Z, grad_x, grad_y = self.compute_gradient_field()
        if Z is not None:
            self.ax.clear()
            cax = self.ax.imshow(Z, origin='lower', aspect='auto', cmap='viridis',
                                 extent=[X.min(), X.max(), Y.min(), Y.max()])
            self.ax.set_title("Concentration Heatmap")
            self.ax.set_xlabel("X (μm)")
            self.ax.set_ylabel("Y (μm)")
            self.canvas.figure.colorbar(cax, ax=self.ax, label="Concentration (wt%)")
            self.canvas.draw()

    def plot_vector_field(self):
        X, Y, Z, grad_x, grad_y = self.compute_gradient_field()
        if Z is not None:
            self.vector_ax.clear()
            cax = self.vector_ax.imshow(Z, origin='lower', aspect='auto', cmap='viridis',
                                       extent=[X.min(), X.max(), Y.min(), Y.max()])
            self.vector_ax.streamplot(X, Y, grad_x, grad_y, color='black', density=1.2)
            self.vector_ax.set_title("Gradient Vector Field ∇C")
            self.vector_ax.set_xlabel("X (μm)")
            self.vector_ax.set_ylabel("Y (μm)")
            self.vector_canvas.figure.colorbar(cax, ax=self.vector_ax, label="Concentration (wt%)")
            self.vector_canvas.draw()

    def analyze_gradient_angles(self):
        if self.gradient_data is not None:
            dominant_angle = self.gradient_data['Angle(deg)'].mean()
            X_feats = self.gradient_data[['dC/dx', 'dC/dy']].values
            try:
                k = int(self.k_input.currentText())
            except:
                k = 2
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X_feats)
            self.gradient_data['Direction Cluster'] = kmeans.labels_

            summary = f"Mean Angle: {dominant_angle:.2f}°"
            for i in range(k):
                avg = self.gradient_data[self.gradient_data['Direction Cluster'] == i]['Angle(deg)'].mean()
                summary += f"Cluster {i} Mean Angle: {avg:.2f}°"
            self.result_label.setText(summary)

    def find_optimal_k(self):
        if self.gradient_data is not None:
            from sklearn.metrics import silhouette_score
            import matplotlib.pyplot as plt

            X_feats = self.gradient_data[['dC/dx', 'dC/dy']].values
            distortions = []
            silhouette_scores = []
            K = range(2, 8)
            for k in K:
                kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
                labels = kmeans.fit_predict(X_feats)
                distortions.append(kmeans.inertia_)
                silhouette_scores.append(silhouette_score(X_feats, labels))

            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].plot(K, distortions, 'bx-')
            ax[0].set_title('Elbow Method')
            ax[0].set_xlabel('k')
            ax[0].set_ylabel('Inertia')

            ax[1].plot(K, silhouette_scores, 'go-')
            ax[1].set_title('Silhouette Scores')
            ax[1].set_xlabel('k')
            ax[1].set_ylabel('Score')

            plt.tight_layout()
            plt.show()

    def export_gradient_csv(self):
        if self.gradient_data is not None:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(self, "Export ∇C CSV", "gradient_vectors.csv", "CSV Files (*.csv);;All Files (*)", options=options)
            if file_name:
                self.gradient_data.to_csv(file_name, index=False)

    def erfc_model(self, x, D, C0, Cb, t):
        return Cb + (C0 - Cb) * erfc(x / (2 * np.sqrt(D * t)))

    def radial_model(self, r, M, D, t):
        return (M / (4 * np.pi * D * t)) * np.exp(-r**2 / (4 * D * t))

    def slab_model(self, x, Cs, D, L, t):
        series_sum = np.sum([
            (4 / np.pi) * (1 / (2 * n + 1)) * np.exp(-(2 * n + 1)**2 * np.pi**2 * D * t / (4 * L**2)) *
            np.sin((2 * n + 1) * np.pi * x / (2 * L)) for n in range(10)
        ], axis=0)
        return Cs * (1 - series_sum)

    def exponential_model(self, x, A, k):
        return A * np.exp(-k * x)

    def fit_exponential(self, x, c):
        x = x - x.min()  # Normalize position
        try:
            popt, _ = curve_fit(self.exponential_model, x, c, p0=[max(c), 0.01])
            fit_vals = self.exponential_model(x, *popt)
            rmse = np.sqrt(np.mean((c - fit_vals) ** 2))
            self.model_rmse["Exponential Decay"] = rmse
            self.result_label.setText(f"Exponential Fit:\nA = {popt[0]:.2f}, k = {popt[1]:.4f}, RMSE = {rmse:.4f}")
        except Exception as e:
            self.result_label.setText("Exponential fit failed: " + str(e))

    def gaussian_2d_model(self, coords, A, x0, y0, D, t):
        x, y = coords
        return A * np.exp(-((x - x0)**2 + (y - y0)**2) / (4 * D * t))

    def fit_gaussian2d(self, x, y, c):
        t = self.get_time_value()
        try:
            popt, _ = curve_fit(self.gaussian_2d_model, (x, y), c,
                                p0=[max(c), np.mean(x), np.mean(y), 1e-5, t])
            fit_vals = self.gaussian_2d_model((x, y), *popt)
            rmse = np.sqrt(np.mean((c - fit_vals) ** 2))
            self.model_rmse["2D Gaussian"] = rmse
            self.result_label.setText(
                f"2D Gaussian Fit:\nD ≈ {popt[3]:.2e}, Center = ({popt[1]:.1f}, {popt[2]:.1f}), RMSE = {rmse:.4f}"
            )
        except Exception as e:
            self.result_label.setText("2D Gaussian fit failed: " + str(e))

    def cylindrical_model(self, r, M, D, t):
        return M / (4 * np.pi * D * t) * np.exp(-r**2 / (4 * D * t))

    def fit_cylindrical(self, x, y, c):
        t = self.get_time_value()

        r = np.sqrt(x**2 + y**2)
        try:
            popt, _ = curve_fit(lambda r, M, D: self.cylindrical_model(r, M, D, t), r, c, p0=[1.0, 1e-5])
            fit_vals = self.cylindrical_model(r, *popt, t)
            rmse = np.sqrt(np.mean((c - fit_vals)**2))
            self.model_rmse["Cylindrical"] = rmse
            self.result_label.setText(f"Cylindrical Fit:\nD ≈ {popt[1]:.2e}, M ≈ {popt[0]:.3f}, RMSE = {rmse:.4f}")
        except Exception as e:
            self.result_label.setText("Cylindrical fit failed: " + str(e))

    def spherical_model(self, r, Cs, R, D, t):
        return Cs * erfc((r - R) / (2 * np.sqrt(D * t)))

    def fit_spherical(self, x, y, c):
        t = self.get_time_value()

        r = np.sqrt(x**2 + y**2)
        try:
            popt, _ = curve_fit(lambda r, Cs, R, D: self.spherical_model(r, Cs, R, D, t),
                                r, c, p0=[max(c), np.mean(r), 1e-5], bounds=(0, np.inf))
            fit_vals = self.spherical_model(r, *popt, t)
            rmse = np.sqrt(np.mean((c - fit_vals)**2))
            self.model_rmse["Spherical"] = rmse
            self.result_label.setText(f"Spherical Fit:\nD ≈ {popt[2]:.2e}, R ≈ {popt[1]:.1f}, RMSE = {rmse:.4f}")
        except Exception as e:
            self.result_label.setText("Spherical fit failed: " + str(e))

    def fit_radial(self, x, y, c):
        r = np.sqrt(x**2 + y**2)
        t = self.get_time_value()

        try:
            popt, _ = curve_fit(lambda r, M, D: self.radial_model(r, M, D, t), r, c, p0=[1, 1e-5])
            fit_vals = self.radial_model(r, *popt, t)
            rmse = np.sqrt(np.mean((c - fit_vals)**2))
            self.model_rmse["2D Radial"] = rmse
            self.result_label.setText(f"2D Radial fit\nD ≈ {popt[1]:.2e}, RMSE = {rmse:.4f}")
        except Exception as e:
            self.result_label.setText("Radial fit failed: " + str(e))

    def fit_slab(self, x, c):
        t = self.get_time_value()

        L = max(x) - min(x)
        try:
            popt, _ = curve_fit(lambda x, Cs, D: self.slab_model(x, Cs, D, L, t), x, c,
                                p0=[max(c), 1e-5], bounds=(0, np.inf))
            fit_vals = self.slab_model(x, *popt, L, t)
            rmse = np.sqrt(np.mean((c - fit_vals)**2))
            self.model_rmse["1D Slab"] = rmse
            self.result_label.setText(f"1D Slab fit\nD ≈ {popt[1]:.2e}, RMSE = {rmse:.4f}")
        except Exception as e:
            self.result_label.setText("Slab fit failed: " + str(e))

    def fit_erfc(self, x, c):
        x = x - x.min()
        t = self.get_time_value()

        try:
            popt, _ = curve_fit(lambda x, D, C0, Cb: self.erfc_model(x, D, C0, Cb, t), x, c,
                                p0=[1e-5, max(c), min(c)], bounds=(0, np.inf))
            fit_vals = self.erfc_model(x, *popt, t)
            rmse = np.sqrt(np.mean((c - fit_vals) ** 2))
            self.model_rmse["1D erfc"] = rmse
            self.result_label.setText(f"1D erfc fit\nD ≈ {popt[0]:.2e}, RMSE = {rmse:.4f}")
        except Exception as e:
            self.result_label.setText("1D erfc fit failed: " + str(e))

    def auto_suggest_model(self, x, y, c):
        self.model_rmse = {}
        self.fit_erfc(x, c)
        self.fit_exponential(x, c)
        self.fit_gaussian2d(x, y, c)
        self.fit_radial(x, y, c)
        self.fit_slab(x, c)
        self.fit_cylindrical(x, y, c)
        self.fit_spherical(x, y, c)

        if self.model_rmse:
            best_model = min(self.model_rmse, key=self.model_rmse.get)
            self.result_label.setText(f"Best Fit: {best_model}\nRMSE = {self.model_rmse[best_model]:.4f}")
        else:
            self.result_label.setText("Model fitting failed for all models.")

    def fit_selected_model(self):
        if self.data is None:
            self.result_label.setText("No data loaded.")
            return
        model = self.model_selector.currentText()
        x = self.data.iloc[:, 0].values
        y = self.data.iloc[:, 1].values
        c = self.data.iloc[:, 2].values

        self.model_rmse = {} # Reset RMSE dict

        if model == "1D erfc":
            self.fit_erfc(x, c)
        elif model == "2D Gaussian":
            self.fit_gaussian2d(x, y, c)
        elif model == "Exponential Decay":
            self.fit_exponential(x, c)
        elif model == "2D Radial":
            self.fit_radial(x, y, c)
        elif model == "1D Slab":
            self.fit_slab(x, c)
        elif model == "Cylindrical":
            self.fit_cylindrical(x, y, c)
        elif model == "Spherical":
            self.fit_spherical(x, y, c)
        elif model == "Auto Suggest":
            self.auto_suggest_model(x, y, c)

    def plot_surface3d(self):
        if self.data is not None:
            self.ax3d.clear()
            x = self.data.iloc[:, 0].values
            y = self.data.iloc[:, 1].values
            c = self.data.iloc[:, 2].values

            df_pivot = pd.DataFrame({'x': x, 'y': y, 'c': c})
            pivot = df_pivot.pivot_table(index='y', columns='x', values='c')
            pivot = pivot.fillna(0)

            X, Y = np.meshgrid(pivot.columns.values, pivot.index.values)
            Z = pivot.values

            self.ax3d.plot_surface(X, Y, Z, cmap='viridis', alpha=0.85, edgecolor='k', linewidth=0.2)
            self.ax3d.set_title("3D Surface Plot ")
            self.ax3d.set_xlabel("X (μm)")
            self.ax3d.set_ylabel("Y (μm)")
            self.ax3d.set_zlabel("Concentration (wt%)")
            self.canvas3d.draw()

    def set_3d_view(self, elev, azim):
        self.ax3d.view_init(elev=elev, azim=azim)
        self.canvas3d.draw()

    def toggle_rotation(self, checked):
        if checked:
            self.rotate_button.setText("Stop Rotation")
            self.start_rotation()
        else:
            self.rotate_button.setText("Start Rotation")
            if hasattr(self, 'timer'):
                self.timer.stop()
                self.timer.deleteLater()
                del self.timer

    def start_rotation(self):
        from PyQt5.QtCore import QTimer
        self.rotation_angle = getattr(self, 'rotation_angle', 0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.rotate_step)
        self.timer.start(100)

    def rotate_step(self):
        self.rotation_angle = (self.rotation_angle + 2) % 360
        self.ax3d.view_init(elev=30, azim=self.rotation_angle)
        self.canvas3d.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = EPMAViewer()
    viewer.show()
    sys.exit(app.exec_())
