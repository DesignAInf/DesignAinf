import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math

class ActiveInferenceUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Active Inference Design Tool")
        self.root.geometry("1400x800")

        # Main layout configuration
        self.root.columnconfigure([0, 1], weight=1)
        self.root.rowconfigure([0, 1, 2], weight=1)

        # Interface Design Section
        self.design_frame = ttk.LabelFrame(root, text="Graphical Interface Design", padding=10)
        self.design_frame.grid(row=0, column=0, rowspan=3, padx=10, pady=10, sticky="nsew")

        self.canvas = tk.Canvas(self.design_frame, width=800, height=600, bg="#f9f9f9", bd=0, highlightthickness=1, highlightbackground="#d3d3d3")
        self.canvas.pack(fill="both", expand=True)

        # Parameters Section
        self.params_frame = ttk.LabelFrame(root, text="Simulation Parameters", padding=10)
        self.params_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.precision_scale = self.create_slider(self.params_frame, "Precision Crafting", 0)
        self.curiosity_scale = self.create_slider(self.params_frame, "Curiosity Sculpting", 1)
        self.prediction_scale = self.create_slider(self.params_frame, "Prediction Embedding", 2)

        # Add shape buttons under parameters
        self.button_frame = ttk.LabelFrame(self.params_frame, text="Shape Tools", padding=10)
        self.button_frame.grid(row=3, column=0, columnspan=2, pady=10, sticky="ew")
        ttk.Button(self.button_frame, text="Add Button", command=self.add_button).pack(side="left", padx=5)
        ttk.Button(self.button_frame, text="Add TextBox", command=self.add_textbox).pack(side="left", padx=5)
        ttk.Button(self.button_frame, text="Add Circle", command=self.add_circle).pack(side="left", padx=5)
        ttk.Button(self.button_frame, text="Add Rectangle", command=self.add_rectangle).pack(side="left", padx=5)

        # Simulation Viewer Section
        self.viewer_frame = ttk.LabelFrame(root, text="Simulation Viewer", padding=10)
        self.viewer_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        self.figure = plt.Figure(figsize=(4, 3), dpi=100)
        self.ax_vfe = self.figure.add_subplot(211)
        self.ax_efe = self.figure.add_subplot(212)
        self.ax_vfe.set_title("VFE Over Time")
        self.ax_vfe.set_xlabel("Time")
        self.ax_vfe.set_ylabel("VFE")
        self.ax_efe.set_title("EFE Over Time")
        self.ax_efe.set_xlabel("Time")
        self.ax_efe.set_ylabel("EFE")

        self.figure_canvas = FigureCanvasTkAgg(self.figure, self.viewer_frame)
        self.figure_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Simulation Output Section
        self.output_frame = ttk.LabelFrame(root, text="Simulation Output", padding=10)
        self.output_frame.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")

        ttk.Button(self.output_frame, text="Run Simulation", command=self.run_simulation).pack(pady=10)

        self.output_text = tk.Text(self.output_frame, height=10, wrap="word", bg="#f1f1f1", bd=0, highlightthickness=1, highlightbackground="#d3d3d3")
        self.output_text.pack(fill="both", expand=True, padx=10, pady=10)

        self.selected_shape = None
        self.offset_x = 0
        self.offset_y = 0

    def create_slider(self, frame, label, row):
        ttk.Label(frame, text=label).grid(row=row, column=0, padx=5, pady=5, sticky="w")
        slider = ttk.Scale(frame, from_=0.1, to=1.0, orient="horizontal")
        slider.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
        return slider

    def add_shape(self, create_func, *coords, **kwargs):
        shape = create_func(*coords, **kwargs)
        self.canvas.tag_bind(shape, "<Button-1>", self.start_drag)
        self.canvas.tag_bind(shape, "<B1-Motion>", self.drag_shape)

    def add_button(self):
        self.add_shape(self.canvas.create_rectangle, 50, 50, 150, 100, fill="#add8e6", tags="button")

    def add_textbox(self):
        self.add_shape(self.canvas.create_rectangle, 200, 50, 400, 100, fill="#d3d3d3", tags="textbox")

    def add_circle(self):
        self.add_shape(self.canvas.create_oval, 300, 200, 400, 300, fill="#90ee90", tags="circle")

    def add_rectangle(self):
        self.add_shape(self.canvas.create_rectangle, 500, 200, 650, 300, fill="#ffa07a", tags="rectangle")

    def start_drag(self, event):
        self.offset_x, self.offset_y = event.x, event.y
        self.selected_shape = self.canvas.find_withtag("current")[0]

    def drag_shape(self, event):
        dx, dy = event.x - self.offset_x, event.y - self.offset_y
        self.offset_x, self.offset_y = event.x, event.y
        self.canvas.move(self.selected_shape, dx, dy)

    def run_simulation(self):
        precision = self.precision_scale.get()
        curiosity = self.curiosity_scale.get()
        prediction = self.prediction_scale.get()

        interaction_results, time_series = self.simulate_user(precision, curiosity, prediction)

        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, interaction_results)
        self.update_simulation_viewer(time_series)

    def simulate_user(self, precision, curiosity, prediction):
        vfe_values, efe_values = [], []
        for t in range(1, 101):
            exploration = curiosity * math.exp(-t / 20)
            prediction_effect = prediction * (1 - math.exp(-t / 10))
            precision_effect = -math.log(precision + 1e-6)

            vfe = precision_effect + exploration - prediction_effect
            efe = exploration * prediction_effect - precision_effect

            vfe_values.append(vfe)
            efe_values.append(efe)

        time_series = {"VFE": vfe_values, "EFE": efe_values}
        results = (
            f"Precision: {precision:.2f}\n"
            f"Curiosity: {curiosity:.2f}\n"
            f"Prediction: {prediction:.2f}\n"
            f"Final VFE: {vfe_values[-1]:.2f}\n"
            f"Final EFE: {efe_values[-1]:.2f}\n"
        )
        return results, time_series

    def update_simulation_viewer(self, time_series):
        self.ax_vfe.clear()
        self.ax_efe.clear()

        self.ax_vfe.plot(time_series["VFE"], label="VFE", color="blue", linewidth=2)
        self.ax_efe.plot(time_series["EFE"], label="EFE", color="green", linewidth=2)

        self.ax_vfe.set_title("VFE Over Time")
        self.ax_efe.set_title("EFE Over Time")

        self.ax_vfe.set_xlabel("Time")
        self.ax_vfe.set_ylabel("VFE")
        self.ax_efe.set_xlabel("Time")
        self.ax_efe.set_ylabel("EFE")

        self.ax_vfe.legend()
        self.ax_efe.legend()

        self.figure_canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = ActiveInferenceUI(root)
    root.mainloop()
