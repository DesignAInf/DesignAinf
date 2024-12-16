
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tkinter import Tk, Label, Entry, Scale, Button, Text, END, HORIZONTAL

# Core Design Tool Framework
class ActiveDesignTool:
    def precision_crafting(self, parameters):
        """Optimize measurable product features."""
        def objective(x):
            return np.sum((np.array(x) - np.array(list(parameters.values()))) ** 2)
        result = minimize(objective, x0=np.zeros(len(parameters)))
        return result.x

    def curiosity_sculpting(self, variability):
        """Introduce uncertainty to stimulate exploration."""
        return np.random.normal(0, variability, 10)

    def prediction_embedding(self, user_goals):
        """Simulate user interaction through active inference."""
        return {goal: np.random.random() for goal in user_goals}

    def generate_design_solution(self, parameters, variability, user_goals):
        """Run all modules to generate a product design scenario."""
        optimized_params = self.precision_crafting(parameters)
        uncertainty = self.curiosity_sculpting(variability)
        predictions = self.prediction_embedding(user_goals)
        return {
            'Optimized Parameters': optimized_params,
            'Curiosity Variability': uncertainty,
            'User Goal Predictions': predictions
        }

# GUI Implementation
class DesignToolGUI:
    def __init__(self, root):
        self.root = root
        self.tool = ActiveDesignTool()
        self.results = None
        self.setup_gui()

    def setup_gui(self):
        self.root.title("Active Inference Design Tool")

        # Precision Crafting Inputs
        Label(self.root, text="Precision Crafting Parameters").grid(row=0, column=0, columnspan=2, pady=5)
        Label(self.root, text="Feature 1:").grid(row=1, column=0, padx=5, sticky="e")
        self.feature1_entry = Entry(self.root)
        self.feature1_entry.grid(row=1, column=1, padx=5)

        Label(self.root, text="Feature 2:").grid(row=2, column=0, padx=5, sticky="e")
        self.feature2_entry = Entry(self.root)
        self.feature2_entry.grid(row=2, column=1, padx=5)

        # Curiosity Sculpting Slider
        Label(self.root, text="Curiosity Sculpting Variability").grid(row=3, column=0, columnspan=2, pady=5)
        self.variability_slider = Scale(self.root, from_=0.1, to=1.0, orient=HORIZONTAL, resolution=0.1)
        self.variability_slider.set(0.5)
        self.variability_slider.grid(row=4, column=0, columnspan=2, pady=5)

        # User Goals Input
        Label(self.root, text="User Goals (comma-separated)").grid(row=5, column=0, columnspan=2, pady=5)
        self.user_goals_entry = Entry(self.root)
        self.user_goals_entry.grid(row=6, column=0, columnspan=2, pady=5)

        # Run Button
        self.run_button = Button(self.root, text="Generate Design Solution", command=self.run_tool)
        self.run_button.grid(row=7, column=0, pady=10)

        # Visualize Button
        self.visualize_button = Button(self.root, text="Visualize Results", command=self.visualize_results)
        self.visualize_button.grid(row=7, column=1, pady=10)

        # Results Display
        self.results_text = Text(self.root, height=10, width=50)
        self.results_text.grid(row=8, column=0, columnspan=2, pady=10)

    def run_tool(self):
        # Collect inputs
        try:
            feature1 = float(self.feature1_entry.get())
            feature2 = float(self.feature2_entry.get())
            variability = self.variability_slider.get()
            user_goals = [goal.strip() for goal in self.user_goals_entry.get().split(',')]

            # Generate solution
            self.results = self.tool.generate_design_solution(
                parameters={'Feature 1': feature1, 'Feature 2': feature2},
                variability=variability,
                user_goals=user_goals
            )

            # Display results
            self.results_text.delete("1.0", END)
            self.results_text.insert(END, "=== Results ===\n")
            self.results_text.insert(END, f"Optimized Parameters: {self.results['Optimized Parameters']}\n")
            self.results_text.insert(END, f"Curiosity Variability: {self.results['Curiosity Variability']}\n")
            self.results_text.insert(END, "User Goal Predictions:\n")
            for goal, prob in self.results['User Goal Predictions'].items():
                self.results_text.insert(END, f"  {goal}: {prob:.3f}\n")
        except ValueError:
            self.results_text.delete("1.0", END)
            self.results_text.insert(END, "Error: Please ensure all inputs are valid numbers or text.")

    def visualize_results(self):
        if not self.results:
            self.results_text.delete("1.0", END)
            self.results_text.insert(END, "Error: Run the tool before visualizing results.")
            return

        optimized_params = self.results['Optimized Parameters']
        curiosity_variability = self.results['Curiosity Variability']
        user_goal_predictions = self.results['User Goal Predictions']

        # Subplot 1: Optimized Parameters
        plt.figure(figsize=(12, 8))

        plt.subplot(3, 1, 1)
        plt.bar(["Feature 1", "Feature 2"], optimized_params, color='skyblue')
        plt.title("Optimized Parameters")
        plt.ylabel("Value")
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Subplot 2: Curiosity Variability
        plt.subplot(3, 1, 2)
        plt.plot(curiosity_variability, marker='o', linestyle='-', color='orange')
        plt.title("Curiosity Variability Samples")
        plt.ylabel("Variability")
        plt.xlabel("Sample Index")
        plt.grid(axis='both', linestyle='--', alpha=0.7)

        # Subplot 3: User Goal Predictions
        plt.subplot(3, 1, 3)
        goals = list(user_goal_predictions.keys())
        probabilities = list(user_goal_predictions.values())
        plt.bar(goals, probabilities, color='lightgreen')
        plt.title("User Goal Predictions")
        plt.ylabel("Probability")
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()

# Run GUI
if __name__ == "__main__":
    root = Tk()
    app = DesignToolGUI(root)
    root.mainloop()
