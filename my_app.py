import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import multiprocessing
import threading
import tkinterdnd2 as tkdnd
from PIL import Image, ImageTk
import subprocess
import os
import tempfile
import atexit
import sys
import json
import webbrowser

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
        self.selected_cpus = tk.IntVar(value=self.num_cpus // 2)
        self.population_size = tk.IntVar(value=16)  # New variable

        self.python_env = sys.executable

        self.base_model_weights_path = "models/best_model_stg.pth"
        self.base_model_args_path = "models/base_model_args.json"
        self.da_adapters_weights_path = "models/lora_best.pth"
        self.da_adapters_args_path = "models/da_adapters_args.json"

        # Header
        header_frame = ttk.Frame(root, padding=(5, 5))
        header_frame.pack(fill="x")
        try:
            ratio = 0.571530503
            logo_width = 192
            logo_image = Image.open("logo.png").resize((logo_width, int(logo_width * ratio)))
            self.logo_photo = ImageTk.PhotoImage(logo_image)
            logo_label = ttk.Label(header_frame, image=self.logo_photo)
            logo_label.pack(side="left")
        except Exception:
            logo_label = ttk.Label(header_frame, text="[Logo - Image not found]")
            logo_label.pack(side="left", padx=10)
        title_label = ttk.Label(header_frame, text="Spike2Pop --- Application !", font=(default_font, 20, "bold"))
        title_label.pack(side="left", padx=20)

        def open_github(event=None):
            webbrowser.open("https://github.com/julienbrandoit/ImplementedPapers")

        github_label = ttk.Label(header_frame, text="GitHub repository - Project Updates", foreground="blue",
                                 cursor="hand2", font=(default_font, 12, "underline"))
        github_label.pack(side="left", padx=10)
        github_label.bind("<Button-1>", open_github)

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

        # Drag & Drop support
        self.root.drop_target_register(tkdnd.DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.drop_csv)

        self.load_config()
        self.update_gpu_checkbox()

    def load_config(self):
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base_dir = os.getcwd()

        self.update_log("==\t==\t==\t==\nSetting up the application from the config file...\n=\t=\t=\t=")

        config_path = os.path.join(base_dir, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config_data = json.load(f)
                self.python_env = config_data.get("python_env", self.python_env)
                self.base_model_weights_path = config_data.get("base_model_weights_path", self.base_model_weights_path)
                self.base_model_args_path = config_data.get("base_model_args_path", self.base_model_args_path)
                self.da_adapters_weights_path = config_data.get("da_adapters_weights_path", self.da_adapters_weights_path)
                self.da_adapters_args_path = config_data.get("da_adapters_args_path", self.da_adapters_args_path)

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
        help_text = (
            "Instructions:\n"
            "1. Load a CSV file\n"
            "2. Select a model (STG or DA)\n"
            "3. Optionally change CPU count and GPU usage\n"
            "4. Set population size\n"
            "5. Run the pipeline\n"
            "6. Save results"
        )
        messagebox.showinfo("Help", help_text)

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.set_csv_file(file_path)

    def drop_csv(self, event):
        file_path = event.data.strip()
        if file_path.lower().endswith(".csv"):
            self.set_csv_file(file_path)
        else:
            messagebox.showerror("Error", "Only CSV files are supported.")

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
            "base_model_weights_path": self.base_model_weights_path,
            "base_model_args_path": self.base_model_args_path,
            "da_adapters_weights_path": self.da_adapters_weights_path,
            "da_adapters_args_path": self.da_adapters_args_path
        }
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base_dir = os.getcwd()
        config_path = os.path.join(base_dir, "config.json")
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
        self.kill_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.DISABLED)
        thread = threading.Thread(target=self.execute_script, daemon=True)
        thread.start()

    def execute_script(self):
        python_exe = self.python_env or sys.executable
        base_path = getattr(sys, 'frozen', False) and sys._MEIPASS or os.path.abspath(".")
        script_path = os.path.join(base_path, "script/main.py")
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
            else:
                self.root.after(0, self.update_log, stripped_line)
        for line in self.process.stderr:
            self.root.after(0, self.update_log, line.strip())
        self.process.wait()
        self.process = None
        self.root.after(0, lambda: self.kill_button.config(state=tk.DISABLED))
        if not signal_received:
            self.root.after(0, lambda: self.save_button.config(state=tk.DISABLED))

    def update_log(self, message):
        self.log_area.config(state=tk.NORMAL)
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.yview(tk.END)
        self.log_area.config(state=tk.DISABLED)

    def kill_process(self):
        if self.process:
            self.process.terminate()
            self.process = None
            self.update_log("Process killed.")
            self.kill_button.config(state=tk.DISABLED)

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

if __name__ == "__main__":
    root = tkdnd.Tk()
    app = CSVApp(root)
    root.mainloop()
