import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import multiprocessing
import threading
from PIL import Image, ImageTk
import subprocess
import os
import tempfile
import atexit
import sys
import json
import webbrowser
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

def get_resource_path(relative_path):
    """ Get the absolute path to a resource in a PyInstaller bundle or script directory. """
    try:
        # PyInstaller creates a temporary folder during execution and stores
        # bundled files there.
        base_path = sys._MEIPASS
    except Exception:
        # When running in normal Python mode (not bundled), use the script directory
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, relative_path)

class CSVApp:
    def __init__(self, root):
        default_font = "Arial"

        self.root = root
        self.root.title("Spike2Pop")
        self.root.geometry("1200x800")
        self.root.configure(background="#e1e1e1")

        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("TFrame", background="#e1e1e1")
        self.style.configure("TLabel", background="#e1e1e1", font=(default_font, 12))
        self.style.configure("TButton", font=(default_font, 12))
        self.style.configure("TRadiobutton", background="#e1e1e1", font=(default_font, 12))

        self.csv_file = None
        self.process = None
        self.result_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
        atexit.register(self.cleanup_temp_file)

        self.selected_model = tk.StringVar(value="")
        self.num_cpus = multiprocessing.cpu_count()
        self.selected_cpus = tk.IntVar(value=self.num_cpus)
        self.population_size = tk.IntVar(value=16)  # New variable

        self.python_env = sys.executable

        # Header
        header_frame = ttk.Frame(root, padding=(5, 5))
        header_frame.pack(fill="x")
        try:
            ratio = 0.571530503
            logo_width = 192
            logo_path = get_resource_path("logo.png")
            if os.path.exists(logo_path):
                logo_image = Image.open(logo_path).resize((logo_width, int(logo_width * ratio)))
                self.logo_photo = ImageTk.PhotoImage(logo_image)
                logo_label = ttk.Label(header_frame, image=self.logo_photo)
                logo_label.pack(side="left")
            else:
                raise FileNotFoundError(f"Logo file not found at {logo_path}")
        except Exception as e:
            print(f"Error loading logo: {e}")
            logo_label = ttk.Label(header_frame, text="[Logo - Image not found]")
            logo_label.pack(side="left", padx=10)
        title_label = ttk.Label(header_frame, text="Spike2Pop --- Application !", font=(default_font, 20, "bold"))
        title_label.pack(side="left", padx=20)

        def open_github(event=None):
            webbrowser.open("https://github.com/julienbrandoit/Spike2Pop---Bridging-Experimental-Neuroscience-and-Computational-Modeling")

        github_label = ttk.Label(header_frame, text="GitHub repository - Project Updates", foreground="blue",
                                 cursor="hand2", font=(default_font, 12, "underline"))
        github_label.pack(side="left", padx=10)
        github_label.bind("<Button-1>", open_github)

        # Add GOODBYE! button
        goodbye_button = ttk.Button(header_frame, text="GOODBYE!", command=root.destroy)
        goodbye_button.pack(side="right", padx=10)

        # Content
        content_frame = ttk.Frame(root, padding=(10, 10))
        content_frame.pack(fill="both", expand=True)
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=2)

        # Left panel
        left_frame = ttk.Frame(content_frame, padding=(10, 10))
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        file_frame = ttk.LabelFrame(left_frame, text="File Selection", padding=(10, 10))
        file_frame.pack(fill="x", pady=5)
        self.load_button = ttk.Button(file_frame, text="Load CSV", command=self.load_csv)
        self.load_button.pack(side="left")
        self.file_label = ttk.Label(file_frame, text="No file selected", foreground="gray")
        self.file_label.pack(side="left", padx=10)

        model_frame = ttk.LabelFrame(left_frame, text="Model Selection", padding=(10, 10))
        model_frame.pack(fill="x", pady=5)
        self.stg_radio = ttk.Radiobutton(model_frame, text="STG model", variable=self.selected_model, value="STG")
        self.stg_radio.pack(side="left", padx=5)
        self.da_radio = ttk.Radiobutton(model_frame, text="DA model", variable=self.selected_model, value="DA")
        self.da_radio.pack(side="left", padx=5)

        cpu_frame = ttk.LabelFrame(left_frame, text="CPU Selection", padding=(10, 10))
        cpu_frame.pack(fill="x", pady=5)
        cpu_label = ttk.Label(cpu_frame, text="Select number of CPUs:")
        cpu_label.pack(side="left")
        self.cpu_selector = ttk.Combobox(cpu_frame, textvariable=self.selected_cpus,
                                         values=list(range(1, self.num_cpus + 1)), state="readonly", width=5)
        self.cpu_selector.pack(side="left", padx=5)
        self.cpu_selector.current(self.selected_cpus.get() - 1)

        # New population size input
        pop_size_frame = ttk.LabelFrame(left_frame, text="Population Size", padding=(10, 10))
        pop_size_frame.pack(fill="x", pady=5)
        pop_label = ttk.Label(pop_size_frame, text="Enter population size:")
        pop_label.pack(side="left")
        pop_entry = ttk.Entry(pop_size_frame, textvariable=self.population_size, width=10)
        pop_entry.pack(side="left", padx=5)

        env_frame = ttk.LabelFrame(left_frame, text="Python Environment", padding=(10, 10))
        env_frame.pack(fill="x", pady=5)
        self.env_button = ttk.Button(env_frame, text="Select Python Env", command=self.select_python_env)
        self.env_button.pack(side="left")
        self.env_label = ttk.Label(env_frame, text=os.path.basename(self.python_env), foreground="gray")
        self.env_label.pack(side="left", padx=10)
        self.save_config_button = ttk.Button(env_frame, text="Save Default Env", command=self.save_config)
        self.save_config_button.pack(side="left", padx=5)

        gpu_frame = ttk.LabelFrame(left_frame, text="GPU Options", padding=(10, 10))
        gpu_frame.pack(fill="x", pady=5)
        self.use_gpu = tk.BooleanVar(value=False)
        self.gpu_checkbox = ttk.Checkbutton(gpu_frame, text="Use GPU", variable=self.use_gpu)
        self.gpu_checkbox.pack(side="left")

        action_frame = ttk.Frame(left_frame, padding=(10, 10))
        action_frame.pack(fill="x", pady=5)
        self.run_button = ttk.Button(action_frame, text="Use the pipeline", command=self.run_pipeline, state=tk.DISABLED)
        self.run_button.pack(side="top", fill="x", pady=2)

        self.sim_panel_button = ttk.Button(action_frame, text="Toward Simulation Panel", command=self.open_simulation_panel, state=tk.DISABLED)
        self.sim_panel_button.pack(side="top", fill="x", pady=2)

        self.kill_button = ttk.Button(action_frame, text="Kill Process", command=self.kill_process, state=tk.DISABLED)
        self.kill_button.pack(side="top", fill="x", pady=2)

        self.save_button = ttk.Button(action_frame, text="Save Results", command=self.save_results, state=tk.DISABLED)
        self.save_button.pack(side="top", fill="x", pady=2)

        help_button = ttk.Button(left_frame, text="Help", command=self.show_help)
        help_button.pack(pady=5, anchor="center")

        # Right panel
        right_frame = ttk.Frame(content_frame, padding=(10, 10))
        right_frame.grid(row=0, column=1, sticky="nsew")
        log_label = ttk.Label(right_frame, text="Log Output:")
        log_label.pack(anchor="w")
        self.log_area = scrolledtext.ScrolledText(right_frame, wrap="word", width=50, height=20,
                                                  state=tk.DISABLED, font=("Courier", 10))
        self.log_area.pack(fill="both", expand=True, pady=5)

        self.load_config()
        self.update_gpu_checkbox()

    def load_config(self):
        self.update_log("==\t==\t==\t==\nSetting up the application from the config file...\n=\t=\t=\t=")

        config_path = get_resource_path("config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config_data = json.load(f)
                self.python_env = config_data.get("python_env", self.python_env)
                if self.python_env == "":
                    self.python_env = sys.executable

                if hasattr(self, "env_label"):
                    self.env_label.config(text=os.path.basename(self.python_env))
                self.update_log(f"Loaded config from {config_path}")
            except Exception as e:
                self.update_log(f"Error loading config: {str(e)}")

        self.update_log("=\t=\t=\t=\nApplication setup complete.\n==\t==\t==\t==")

    def check_gpu_availability(self):
        try:
            result = subprocess.run(
                [self.python_env, "-c", "import torch; print(torch.cuda.is_available())"],
                capture_output=True, text=True, timeout=10
            )
            return result.stdout.strip() == "True"
        except Exception:
            return False

    def update_gpu_checkbox(self):
        available = self.check_gpu_availability()
        if available:
            self.use_gpu.set(True)
            self.gpu_checkbox.config(state=tk.NORMAL)
            self.update_log("\nCUDA is available. 'Use GPU' enabled and checked.\n")
        else:
            self.use_gpu.set(False)
            self.gpu_checkbox.config(state=tk.DISABLED)
            self.update_log("\nCUDA not available or PyTorch not installed. 'Use GPU' disabled.\n")

    def show_help(self):
        help_window = tk.Toplevel(self.root)
        help_window.title("Help - Instructions")
        help_window.geometry("600x400")  # You can adjust size here
        help_window.resizable(True, True)

        text_widget = scrolledtext.ScrolledText(help_window, wrap="word", font=("Arial", 11))
        text_widget.pack(expand=True, fill="both", padx=10, pady=10)

        help_text = (
            "Spike2Pop - Quick Start Guide\n\n"
            "1. Load a CSV file with at least two columns:\n"
            "   - 'spiking_time': contains spike time data\n"
            "The format for the spike time sequences should be \"[3045.0, ..., 4444.0]\" (a string with double quotes, square brackets, and comma-separated values).\n"
            "   - 'ID': uniquely identifies each sequence or trial\n\n"
            "2. Select a model:\n"
            "   - STG: stomatogastric ganglion model\n"
            "   - DA: dopaminergic neuron model\n\n"
            "3. Configure settings:\n"
            "   - Choose number of CPUs to use\n"
            "   - Set the desired population size\n"
            "   - (Optional) Enable GPU if available\n\n"
            "4. Click 'Use the pipeline' to start the process\n"
            "5. After completion, use 'Save Results' to export data\n"
            "The results is a CSV file that contains the generated population. Each instances is defined by its conductances values along with the corresponding ID.\n\n"

            "Simulation Panel:\n\n"
            "Once you have generated the population, you can proceed to the simulation panel:\n"
            "1. Click 'Toward Simulation Panel' to open the simulation panel.\n"
            "2. Set the simulation duration and step size.\n"
            "3. Select the IDs you want to simulate by checking the corresponding checkboxes.\n"
            "4. Click 'SIMULATE!' to start the simulation.\n"
            "5. After the simulation is complete, you can view the results by clicking 'See Results' for each ID. The trace values are stored along the conductance values for the simulated instances.\n\n"

            "\nIf you encounter any issues, please refer to the GitHub repository for documentation and updates.\n"
        )
        text_widget.insert(tk.END, help_text)
        text_widget.config(state=tk.DISABLED)

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.set_csv_file(file_path)

    def set_csv_file(self, file_path):
        self.csv_file = file_path
        self.file_label.config(text=f"Selected: {file_path}")
        self.run_button.config(state=tk.NORMAL)

    def select_python_env(self):
        env_path = filedialog.askopenfilename(
            title="Select Python Interpreter",
            filetypes=[("Python Executable", "python*"), ("All files", "*.*")]
        )
        if env_path:
            self.python_env = env_path
            self.env_label.config(text=os.path.basename(env_path))
            self.update_log(f"Selected Python environment: {env_path}")
            self.update_gpu_checkbox()

    def save_config(self):
        config_data = {
            "python_env": self.python_env,
            "base_model_weights_path": "models/best_model_stg.pth",
            "base_model_args_path": "models/base_model_args.json",
            "base_model_v_th": -51.0,

            "da_adapters_weights_path": "models/lora_best.pth",
            "da_adapters_args_path": "models/lora_args.json",
            "da_adapters_v_th": -55.5
        }
        config_path = get_resource_path("config.json")
        try:
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=4)
            messagebox.showinfo("Config Saved", f"Configuration saved to:\n{config_path}")
            self.update_log(f"Configuration saved to {config_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config: {str(e)}")
            self.update_log(f"Failed to save config: {str(e)}")

    def run_pipeline(self):
        if not self.csv_file or not self.selected_model.get():
            messagebox.showerror("Error", "Please select a CSV file and a model.")
            return

        self.update_log("Running pipeline...")
        self.run_button.config(state=tk.DISABLED)  # Disable run button
        self.kill_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.DISABLED)
        thread = threading.Thread(target=self.execute_script, daemon=True)
        thread.start()

    def execute_script(self):
        python_exe = self.python_env or sys.executable
        script_path = get_resource_path("script/main_script.py")
        command = [
            python_exe,
            script_path,
            self.csv_file,
            self.selected_model.get(),
            str(self.selected_cpus.get()),
            self.result_file,
            str(self.use_gpu.get()),
            str(self.population_size.get())  # <-- Added here
        ]
        self.process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        signal_received = False
        for line in self.process.stdout:
            stripped_line = line.strip()
            if stripped_line == "RESULTS_READY":
                signal_received = True
                self.root.after(0, lambda: self.save_button.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.sim_panel_button.config(state=tk.NORMAL))
            else:
                self.root.after(0, self.update_log, stripped_line)
        for line in self.process.stderr:
            self.root.after(0, self.update_log, line.strip())
        self.process.wait()
        self.process = None
        self.root.after(0, lambda: self.kill_button.config(state=tk.DISABLED))
        if not signal_received:
            self.root.after(0, lambda: self.save_button.config(state=tk.DISABLED))
        self.root.after(0, lambda: self.run_button.config(state=tk.NORMAL))  # Re-enable run button

    def kill_process(self):
        if self.process:
            self.process.terminate()
            self.process = None
            self.update_log("Process killed.")
            self.kill_button.config(state=tk.DISABLED)
            self.run_button.config(state=tk.NORMAL)  # Re-enable run button

    def update_log(self, message):
        self.log_area.config(state=tk.NORMAL)
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.yview(tk.END)
        self.log_area.config(state=tk.DISABLED)

    def save_results(self):
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if save_path and os.path.exists(self.result_file):
            import shutil
            shutil.copy(self.result_file, save_path)
            messagebox.showinfo("Success", "Results saved successfully.")
        else:
            messagebox.showerror("Error", "No result file found to save.")

    def cleanup_temp_file(self):
        if os.path.exists(self.result_file):
            os.remove(self.result_file)

    def open_simulation_panel(self):
        panel = tk.Toplevel(self.root)
        self.sim_panel = panel
        panel.title("Simulation Panel")
        panel.geometry("500x600")

        default_font = ("Arial", 12)

        # Title
        neuron_type = self.selected_model.get()
        if neuron_type == "STG":
            neuron_type = "Stomatogastric Ganglion Neuron"
        elif neuron_type == "DA":
            neuron_type = "Dopaminergic Neuron"
        else:
            raise ValueError("Invalid neuron type selected.")

        title_label = ttk.Label(panel, text=f"Simulation Panel --- {neuron_type}", font=("Arial", 16, "bold"))
        title_label.pack(pady=(10, 10))

        # Simulation Duration
        duration_frame = ttk.Frame(panel, padding=10)
        duration_frame.pack(fill="x")
        ttk.Label(duration_frame, text="Simulation duration (ms):", font=default_font).pack(side="left")
        self.sim_duration = tk.DoubleVar(value=4000.0)
        ttk.Entry(duration_frame, textvariable=self.sim_duration, width=10).pack(side="left", padx=10)

        # Simulation Transitient Duration
        duration_frame_transient = ttk.Frame(panel, padding=10)
        duration_frame_transient.pack(fill="x")
        ttk.Label(duration_frame_transient, text="Transient duration [will be removed] (ms):", font=default_font).pack(side="left")
        self.sim_duration_trans = tk.DoubleVar(value=2000.0)
        ttk.Entry(duration_frame_transient, textvariable=self.sim_duration_trans, width=10).pack(side="left", padx=10)


        # Step Size
        step_frame = ttk.Frame(panel, padding=10)
        step_frame.pack(fill="x")
        ttk.Label(step_frame, text="Step size (ms):", font=default_font).pack(side="left")
        self.step_size = tk.DoubleVar(value=0.05)
        ttk.Entry(step_frame, textvariable=self.step_size, width=10).pack(side="left", padx=10)

        # Horizontal line
        ttk.Separator(panel, orient="horizontal").pack(fill="x", pady=10)

        # Check/Uncheck all buttons
        check_frame = ttk.Frame(panel, padding=10)
        check_frame.pack(fill="x")

        ttk.Button(check_frame, text="Check All", command=self.check_all).pack(side="left", padx=5)
        ttk.Button(check_frame, text="Uncheck All", command=self.uncheck_all).pack(side="left", padx=5)

        # Save Full Traces Checkbox (to the left of SIMULATE!)
        self.save_full_traces = tk.BooleanVar(value=False)
        self.save_full_traces_checkbox = ttk.Checkbutton(
            check_frame,
            text="Save full traces",
            variable=self.save_full_traces
        )
        self.save_full_traces_checkbox.pack(side="right", padx=10)

        # Simulate button
        simulate_button = ttk.Button(check_frame, text="SIMULATE!", command=self.simulate)
        simulate_button.pack(side="right", padx=10)

        # Horizontal line
        ttk.Separator(panel, orient="horizontal").pack(fill="x", pady=10)

        # Scrollable frame placeholder
        scroll_canvas = tk.Canvas(panel)
        scroll_frame = ttk.Frame(scroll_canvas)
        scrollbar = ttk.Scrollbar(panel, orient="vertical", command=scroll_canvas.yview)
        scroll_canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        scroll_canvas.pack(side="left", fill="both", expand=True)
        scroll_canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

        def on_configure(event):
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))

        scroll_frame.bind("<Configure>", on_configure)

        def on_mousewheel(event):
            scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        scroll_canvas.bind_all("<MouseWheel>", on_mousewheel)  # Windows and macOS
        scroll_canvas.bind_all("<Button-4>", lambda e: scroll_canvas.yview_scroll(-1, "units"))  # Linux scroll up
        scroll_canvas.bind_all("<Button-5>", lambda e: scroll_canvas.yview_scroll(1, "units"))   # Linux scroll down

        # we load the 'ID' column from the output CSV file
        try:
            df = pd.read_csv(self.result_file, usecols=['ID'])
            # number of time each id appears in the result file
            ids_count = df['ID'].value_counts()
            ids = df['ID'].unique()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load result file: {str(e)}")
            # close the simulation panel
            panel.destroy()
            return

        # Create checkbuttons for each ID
        self.checkbuttons = []
        self.see_results_buttons = []  # To store references to the "See Results" buttons
        for idx, id_ in enumerate(ids):
            id_size_ = ids_count[id_]
            var = tk.BooleanVar(value=True)
            checkbutton = ttk.Checkbutton(scroll_frame, text=f"ID [{id_size_} instances]: {id_}", variable=var)
            checkbutton.grid(row=idx, column=0, sticky="w", padx=10, pady=2)
            self.checkbuttons.append((id_, var))

            # Create "See Results" button and align it with the checkbox
            see_results_button = ttk.Button(
                scroll_frame,
                text=f"See Results ({id_})",
                command=lambda id_=id_: self.display_results(id_),
                state=tk.DISABLED
            )
            see_results_button.grid(row=idx, column=1, padx=10, pady=2)
            self.see_results_buttons.append(see_results_button)

        # Save references to checkbuttons for "check all" logic
        self.sim_checkboxes = scroll_frame.winfo_children()
        self.sim_button_exe = simulate_button

    def check_all(self):
        for cb in getattr(self, 'sim_checkboxes', []):
            cb.state(['!alternate', 'selected'])
        for id_, var in self.checkbuttons:
            var.set(True)
        self.sim_panel.update_idletasks()
        self.sim_panel.update()

    def uncheck_all(self):
        for cb in getattr(self, 'sim_checkboxes', []):
            cb.state(['!alternate', '!selected'])
        for id_, var in self.checkbuttons:
            var.set(False)
        self.sim_panel.update_idletasks()
        self.sim_panel.update()

    def simulate(self):
        selected_ids = [id_ for id_, var in self.checkbuttons if var.get()]
        selected_ids_str = ",".join(map(str, selected_ids))

        if not selected_ids:
            messagebox.showwarning("No Selection", "Please select at least one ID for simulation.")
            return

        # get simulation parameters
        try:
            self.sim_duration = float(self.sim_duration.get())
            self.step_size = float(self.step_size.get())
            self.sim_duration_trans = float(self.sim_duration_trans.get())
            self.save_full_traces_value = self.save_full_traces.get()
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers for simulation duration and step size.")
            return
        if self.sim_duration <= 0 or self.step_size <= 0 or self.sim_duration_trans <= 0:
            messagebox.showerror("Invalid Input", "Simulation duration, step size and transient duration must be positive.")
            return
        if self.sim_duration < self.sim_duration_trans:
            messagebox.showerror("Invalid Input", "Simulation duration must be greater than transient duration.")
            return
        # Check if the selected model is valid
        if self.selected_model.get() not in ["STG", "DA"]:
            messagebox.showerror("Invalid Model", "Please select a valid model (STG or DA).")
            return

        # Store parameters for the thread
        self.sim_args = {
            "neuron_type": self.selected_model.get(),
            "num_cpus": self.selected_cpus.get(),
            "output_file": self.result_file,
            "selected_ids": selected_ids_str,
            "sim_duration": self.sim_duration,
            "sim_duration_trans": self.sim_duration_trans,
            "step_size": self.step_size,
            "save_full_traces": self.save_full_traces_value
        }

        #self.root.after(0, lambda: self.sim_panel.destroy())
        self.root.after(0, lambda: self.sim_panel_button.config(state=tk.DISABLED))
        self.root.after(0, lambda: self.kill_button.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.save_button.config(state=tk.DISABLED))
        self.root.after(0, lambda: self.run_button.config(state=tk.DISABLED))
        #disable the save full traces checkbox
        self.root.after(0, lambda: self.save_full_traces_checkbox.state(['disabled']))
        
        self.sim_button_exe.config(state=tk.DISABLED)

        thread = threading.Thread(target=self.execute_simulation, daemon=True)
        thread.start()

    def display_results(self, id_):
        # Create a new window to display the plots
        plot_window = tk.Toplevel(self.root)
        plot_window.title(f"Results for ID: {id_}")
        plot_window.geometry("1200x600")

        # Create a scrollable frame to hold the plots
        scroll_canvas = tk.Canvas(plot_window)
        scroll_frame = ttk.Frame(scroll_canvas)
        scrollbar = ttk.Scrollbar(plot_window, orient="vertical", command=scroll_canvas.yview)
        scroll_canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        scroll_canvas.pack(side="left", fill="both", expand=True)
        scroll_canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

        def on_configure(event):
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))

        scroll_frame.bind("<Configure>", on_configure)

        def on_mousewheel(event):
            scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        scroll_canvas.bind_all("<MouseWheel>", on_mousewheel)  # Windows and macOS
        scroll_canvas.bind_all("<Button-4>", lambda e: scroll_canvas.yview_scroll(-1, "units"))  # Linux scroll up
        scroll_canvas.bind_all("<Button-5>", lambda e: scroll_canvas.yview_scroll(1, "units"))   # Linux scroll down

        t_eval = np.arange(self.sim_duration_trans, self.sim_duration + self.step_size, self.step_size)

        if self.selected_model.get() == "STG":
            conductances = ['g_Na', 'g_Kd', 'g_CaT', 'g_CaS', 'g_KCa', 'g_A', 'g_H', 'g_leak']
        elif self.selected_model.get() == "DA":
            conductances = ['g_Na', 'g_Kd', 'g_CaL', 'g_CaN', 'g_ERG', 'g_NMDA', 'g_leak']
        else:
            messagebox.showerror("Invalid Model", "Please select a valid model (STG or DA).")
            return

        try:
            df = pd.read_csv(self.result_file)
            # Filter the DataFrame for the selected ID
            filtered_df = df[df['ID'] == id_].dropna(axis=1, how='all')
            if filtered_df.empty:
                messagebox.showerror("No Data", f"No data found for ID: {id_}")
                return
            if self.save_full_traces_value:
                V = filtered_df['simulation_V']
                if V.empty:
                    messagebox.showerror("No Data", f"No simulation data found for ID: {id_}")
                    return
            else:
                V = filtered_df['spiking_times']
                if V.empty:
                    messagebox.showerror("No Data", f"No spiking time data found for ID: {id_}")
                    return


            # Also load the original data
            original_df = pd.read_csv(self.csv_file)
            original_df = original_df[original_df['ID'] == id_].dropna(axis=1, how='all')
            if original_df.empty:
                messagebox.showerror("No Data", f"No original data found for ID: {id_}")
                return
            original_sp = original_df['spiking_times']
            if original_sp.empty:
                messagebox.showerror("No Data", f"No spiking time data found for ID: {id_}")
                return

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load result file: {str(e)}")
            return

        s = len(V)

        # Parse back into list of float
        V = [np.fromstring(v[1:-1], sep=',') for v in V]
        original_sp = [np.fromstring(sp[1:-1], sep=',') for sp in original_sp]

        original_sp = np.asarray(original_sp)[0]
        original_sp = original_sp - original_sp[0] + self.sim_duration_trans

        # Create a plot for the original spiking times and boxplot of scaled conductances
        fig_original, (ax_original, ax_boxplot) = plt.subplots(1, 2, figsize=(12, 3), gridspec_kw={'width_ratios': [2, 1]})

        # Plot original spiking times
        ax_original.eventplot(original_sp, lineoffsets=0, linelengths=0.5, color="red", label="Original Spiking Times")
        ax_original.set_title(f"Original Recording")
        ax_original.set_xlabel("Time (ms)")
        ax_original.set_ylabel("")
        ax_original.set_xlim(self.sim_duration_trans, self.sim_duration)
        ax_original.set_ylim(-0.5, 0.5)

        # Calculate global max of all conductances across all instances
        global_max = [filtered_df[cond].astype(float).max() for cond in conductances]
        global_max = np.asarray(global_max)

        # Prepare data for boxplot
        scaled_conductances = []
        for cond in conductances:
            raw_vals = filtered_df[cond].astype(float).values
            scaled_vals = raw_vals / global_max[conductances.index(cond)]
            scaled_conductances.append(scaled_vals)

        # Plot horizontal boxplot of scaled conductances
        bp = ax_boxplot.boxplot(scaled_conductances, vert=False, labels=[f"{cond} ÷ {global_max[i]:.2f}" for i, cond in enumerate(conductances)], showfliers=False)


        # Define a list of colors for the conductances
        conductance_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']


        # Add scatter plot with jitter
        for i, cond in enumerate(conductances):
            y = np.random.normal(i + 1, 0.04, size=len(scaled_conductances[i]))
            ax_boxplot.scatter(scaled_conductances[i], y, color=conductance_colors[i], alpha=0.5, s=10)

        ax_boxplot.set_title("Boxplot of Scaled Conductances")
        ax_boxplot.set_xlabel("Scaled Values")
        ax_boxplot.set_xlim(0, 1.1)

        fig_original.tight_layout()
        canvas_original = FigureCanvasTkAgg(fig_original, master=scroll_frame)
        canvas_original.draw()
        canvas_original.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=10)

        # Separator after original plot
        separator = ttk.Separator(scroll_frame, orient='horizontal')
        separator.pack(fill='x', pady=10)

        for i in range(s):
            # Create a figure with two subplots: voltage trace and conductance barplot
            fig_sim, (ax_trace, ax_bar) = plt.subplots(1, 2, figsize=(12, 3), gridspec_kw={'width_ratios': [2, 1]})

            # Plot the voltage trace
            if self.save_full_traces_value:
                ax_trace.plot(t_eval, V[i], label=f"Simulation {i+1}", color="blue", linewidth=1)
                ax_trace.set_title(f"Instance {i+1}")
                ax_trace.set_xlabel("Time (ms)")
                ax_trace.set_ylabel("Voltage (mV)")
                ax_trace.set_xlim(self.sim_duration_trans, self.sim_duration)
            else:
                ax_trace.eventplot(V[i], lineoffsets=0, linelengths=0.5, color="blue", label=f"Simulation {i+1}")
                ax_trace.set_title(f"Spiking Times for Instance {i+1}")
                ax_trace.set_xlabel("Time (ms)")
                ax_trace.set_ylabel("")
                ax_trace.set_xlim(self.sim_duration_trans, self.sim_duration)

            # Prepare and scale conductance values
            raw_vals = [float(filtered_df[cond].iloc[i]) for cond in conductances]

            raw_vals = np.asarray(raw_vals)

            scaled_vals = raw_vals/global_max

            # Plot conductances as a horizontal bar plot
            ax_bar.barh(conductances, scaled_vals, color=conductance_colors[:len(conductances)])
            ax_bar.set_title("Conductances")
            ax_bar.set_xlim(0, 1)
            ax_bar.set_xlabel(f"Values (mS/cm²)")

            # yticks should be 'g_i x scaling_factor_i
            ytick_labels = [f"{cond} ÷ {global_max[i]:.2f}" for i, cond in enumerate(conductances)]
            ax_bar.set_yticks(np.arange(len(conductances)))
            ax_bar.set_yticklabels(ytick_labels)
            ax_bar.set_ylim(-0.5, len(conductances) - 0.5)

            # Render the plot in Tkinter
            fig_sim.tight_layout()
            canvas_sim = FigureCanvasTkAgg(fig_sim, master=scroll_frame)
            canvas_sim.draw()
            canvas_sim.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=10)

            # Add a separator
            separator = ttk.Separator(scroll_frame, orient='horizontal')
            separator.pack(fill='x', pady=10)


    def display_results__temp(self, id_):
        # Create a new window to display the plots
        plot_window = tk.Toplevel(self.root)
        plot_window.title(f"Results for ID: {id_}")
        plot_window.geometry("1200x600")

        # Create a scrollable frame to hold the plots
        scroll_canvas = tk.Canvas(plot_window)
        scroll_frame = ttk.Frame(scroll_canvas)
        scrollbar = ttk.Scrollbar(plot_window, orient="vertical", command=scroll_canvas.yview)
        scroll_canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        scroll_canvas.pack(side="left", fill="both", expand=True)
        scroll_canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

        def on_configure(event):
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))

        scroll_frame.bind("<Configure>", on_configure)

        def on_mousewheel(event):
            scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        scroll_canvas.bind_all("<MouseWheel>", on_mousewheel)  # Windows and macOS
        scroll_canvas.bind_all("<Button-4>", lambda e: scroll_canvas.yview_scroll(-1, "units"))  # Linux scroll up
        scroll_canvas.bind_all("<Button-5>", lambda e: scroll_canvas.yview_scroll(1, "units"))   # Linux scroll down

        t_eval = np.arange(self.sim_duration_trans, self.sim_duration + self.step_size, self.step_size)

        if self.selected_model.get() == "STG":
            conductances = ['g_Na', 'g_Kd', 'g_CaT', 'g_CaS', 'g_KCa', 'g_A', 'g_H', 'g_leak']
        elif self.selected_model.get() == "DA":
            conductances = ['g_Na', 'g_Kd', 'g_CaL', 'g_CaN', 'g_ERG', 'g_NMDA', 'g_leak']
        else:
            messagebox.showerror("Invalid Model", "Please select a valid model (STG or DA).")
            return

        try:
            df = pd.read_csv(self.result_file)
            filtered_df = df[df['ID'] == id_].dropna(axis=1, how='all')
            if filtered_df.empty:
                messagebox.showerror("No Data", f"No data found for ID: {id_}")
                return
            V = filtered_df['simulation_V']

            # Also load the original data
            original_df = pd.read_csv(self.csv_file)
            original_df = original_df[original_df['ID'] == id_].dropna(axis=1, how='all')
            if original_df.empty:
                messagebox.showerror("No Data", f"No original data found for ID: {id_}")
                return
            original_sp = original_df['spiking_times']
            if original_sp.empty:
                messagebox.showerror("No Data", f"No spiking time data found for ID: {id_}")
                return

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load result file: {str(e)}")
            return

        s = len(V)

        # Parse original spiking times into array
        original_sp = [np.fromstring(sp[1:-1], sep=',') for sp in original_sp]
        original_sp = np.asarray(original_sp)[0]
        original_sp = original_sp - original_sp[0] + self.sim_duration_trans

        # Plot original spiking times and boxplot of conductances
        fig_original, (ax_original, ax_boxplot) = plt.subplots(1, 2, figsize=(12, 3), gridspec_kw={'width_ratios': [2, 1]})

        ax_original.eventplot(original_sp, lineoffsets=0, linelengths=0.5, color="red", label="Original Spiking Times")
        ax_original.set_title(f"Original Recording")
        ax_original.set_xlabel("Time (ms)")
        ax_original.set_ylabel("")
        ax_original.set_xlim(self.sim_duration_trans, self.sim_duration)
        ax_original.set_ylim(-0.5, 0.5)

        global_max = [filtered_df[cond].astype(float).max() for cond in conductances]
        global_max = np.asarray(global_max)

        scaled_conductances = []
        for cond in conductances:
            raw_vals = filtered_df[cond].astype(float).values
            scaled_vals = raw_vals / global_max[conductances.index(cond)]
            scaled_conductances.append(scaled_vals)

        bp = ax_boxplot.boxplot(scaled_conductances, vert=False,
                            labels=[f"{cond} ÷ {global_max[i]:.2f}" for i, cond in enumerate(conductances)],
                            showfliers=False)

        conductance_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

        for i, cond in enumerate(conductances):
            y = np.random.normal(i + 1, 0.04, size=len(scaled_conductances[i]))
            ax_boxplot.scatter(scaled_conductances[i], y, color=conductance_colors[i], alpha=0.5, s=10)

        ax_boxplot.set_title("Boxplot of Scaled Conductances")
        ax_boxplot.set_xlabel("Scaled Values")
        ax_boxplot.set_xlim(0, 1.1)

        fig_original.tight_layout()
        canvas_original = FigureCanvasTkAgg(fig_original, master=scroll_frame)
        canvas_original.draw()
        canvas_original.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=10)

        separator = ttk.Separator(scroll_frame, orient='horizontal')
        separator.pack(fill='x', pady=10)

        # Now iterate over instances and plot voltage trace or spiking times based on save_full_traces_value
        for i in range(s):
            if self.save_full_traces_value:
                # Parse voltage trace array if possible, else empty array
                v_str = V.iloc[i] if hasattr(V, 'iloc') else V[i]
                V_i = np.fromstring(v_str[1:-1], sep=',') if isinstance(v_str, str) and len(v_str) > 2 else np.array([])

                if V_i.size > 0:
                    fig_sim, (ax_trace, ax_bar) = plt.subplots(1, 2, figsize=(12, 3), gridspec_kw={'width_ratios': [2, 1]})

                    ax_trace.plot(t_eval, V_i, label=f"Simulation {i+1}", color="blue", linewidth=1)
                    ax_trace.set_title(f"Instance {i+1}")
                    ax_trace.set_xlabel("Time (ms)")
                    ax_trace.set_ylabel("Voltage (mV)")
                    ax_trace.set_xlim(self.sim_duration_trans, self.sim_duration)

                    raw_vals = [float(filtered_df[cond].iloc[i]) for cond in conductances]
                    raw_vals = np.asarray(raw_vals)
                    scaled_vals = raw_vals / global_max

                    ax_bar.barh(conductances, scaled_vals, color=conductance_colors[:len(conductances)])
                    ax_bar.set_title("Conductances")
                    ax_bar.set_xlim(0, 1)
                    ax_bar.set_xlabel(f"Values (mS/cm²)")

                    ytick_labels = [f"{cond} ÷ {global_max[j]:.2f}" for j, cond in enumerate(conductances)]
                    ax_bar.set_yticks(np.arange(len(conductances)))
                    ax_bar.set_yticklabels(ytick_labels)
                    ax_bar.set_ylim(-0.5, len(conductances) - 0.5)

                    fig_sim.tight_layout()
                    canvas_sim = FigureCanvasTkAgg(fig_sim, master=scroll_frame)
                    canvas_sim.draw()
                    canvas_sim.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=10)

                    separator = ttk.Separator(scroll_frame, orient='horizontal')
                    separator.pack(fill='x', pady=10)
                    continue  # skip spiking times plot for this instance if trace is shown

            # If full traces not saved or empty trace, plot spiking times eventplot
            spiking_times_str = filtered_df['spiking_times'].iloc[i]
            spiking_times_arr = np.fromstring(spiking_times_str[1:-1], sep=',') if isinstance(spiking_times_str, str) else np.array([])

            if spiking_times_arr.size > 0:
                spiking_times_arr = spiking_times_arr - spiking_times_arr[0] + self.sim_duration_trans

            fig_sp, ax_sp = plt.subplots(figsize=(12, 2))
            ax_sp.eventplot(spiking_times_arr, lineoffsets=0, linelengths=0.5, color='blue', label=f"Spiking Times Instance {i+1}")
            ax_sp.set_title(f"Spiking Times Instance {i+1} (No full trace saved)")
            ax_sp.set_xlabel("Time (ms)")
            ax_sp.set_yticks([])
            ax_sp.set_xlim(self.sim_duration_trans, self.sim_duration)

            fig_sp.tight_layout()
            canvas_sp = FigureCanvasTkAgg(fig_sp, master=scroll_frame)
            canvas_sp.draw()
            canvas_sp.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=10)

            separator = ttk.Separator(scroll_frame, orient='horizontal')
            separator.pack(fill='x', pady=10)


    def execute_simulation(self):
        args = self.sim_args
        python_exe = self.python_env or sys.executable
        script_path = get_resource_path("script/main_simulation.py")
        command = [
            python_exe,
            script_path,
            args["output_file"],
            args["neuron_type"],
            str(args["num_cpus"]),
            args["output_file"],
            args["selected_ids"],
            str(args["sim_duration"]),
            str(args["step_size"]),
            str(args["sim_duration_trans"]),
            str(args["save_full_traces"])
        ]

        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        signal_received = False

        self.update_log("Running simulation...")

        for line in self.process.stdout:
            stripped_line = line.strip()
            if stripped_line == "RESULTS_READY":
                signal_received = True
                self.root.after(0, lambda: self.save_button.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.sim_panel_button.config(state=tk.NORMAL))
                # Enable "See Results" buttons for selected IDs
                for idx, (id_, var) in enumerate(self.checkbuttons):
                    if var.get():
                        self.see_results_buttons[idx].config(state=tk.NORMAL)
            else:
                self.root.after(0, self.update_log, stripped_line)

        for line in self.process.stderr:
            self.root.after(0, self.update_log, line.strip())

        self.process.wait()
        self.process = None
        self.root.after(0, lambda: self.kill_button.config(state=tk.DISABLED))
        if not signal_received:
            self.root.after(0, lambda: self.save_button.config(state=tk.DISABLED))
        self.root.after(0, lambda: self.run_button.config(state=tk.NORMAL))
        
if __name__ == "__main__":
    root = tk.Tk()
    app = CSVApp(root)
    root.mainloop()
    print("Application closed.")
    print("Cleaning up temporary files.")
    app.cleanup_temp_file()
    print("Temporary files cleaned up.")
    print("Goodbye!")