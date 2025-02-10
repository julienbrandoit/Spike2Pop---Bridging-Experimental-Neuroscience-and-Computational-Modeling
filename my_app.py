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
import json  # For handling config file
import webbrowser # For opening links

class CSVApp:
    def __init__(self, root):

        default_font = "Arial"

        self.root = root
        self.root.title("TO DO - FIND A NICE NAME !")
        self.root.geometry("1200x700")
        self.root.configure(background="#e1e1e1")  # light gray background

        # Setup ttk style for a modern look
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("TFrame", background="#e1e1e1")
        self.style.configure("TLabel", background="#e1e1e1", font=(default_font, 12))
        self.style.configure("TButton", font=(default_font, 12))
        self.style.configure("TRadiobutton", background="#e1e1e1", font=(default_font, 12))

        # Internal variables
        self.csv_file = None
        self.process = None  # reference to the subprocess
        self.result_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
        atexit.register(self.cleanup_temp_file)  # cleanup temporary file on exit

        self.selected_model = tk.StringVar(value="")
        self.num_cpus = multiprocessing.cpu_count()
        self.selected_cpus = tk.IntVar(value=self.num_cpus // 2)

        # Initialize variable to store selected interpreter path; default to the running interpreter.
        self.python_env = sys.executable

        # ----------------------------
        # Header: Logo and Title
        # ----------------------------
        header_frame = ttk.Frame(root, padding=(5, 5))
        header_frame.pack(fill="x")
        try:
            ratio = 0.571530503
            logo_width = 192  # adjust the size as needed
            logo_image = Image.open("logo.png").resize((logo_width, int(logo_width * ratio)))
            self.logo_photo = ImageTk.PhotoImage(logo_image)
            logo_label = ttk.Label(header_frame, image=self.logo_photo)
            logo_label.pack(side="left")
        except Exception as e:
            logo_label = ttk.Label(header_frame, text="[Logo - Image not found]")
            logo_label.pack(side="left", padx=10)
        title_label = ttk.Label(header_frame, text="TO DO - FIND A NICE NAME !", font=(default_font, 20, "bold"))
        title_label.pack(side="left", padx=20)

        def open_github(event=None):
                webbrowser.open("https://github.com/julienbrandoit/ImplementedPapers")
            
        github_label = ttk.Label(header_frame, text="GitHub repository - Project Updates", foreground="blue", cursor="hand2", font=(default_font, 12, "underline"))
        github_label.pack(side="left", padx=10)
        github_label.bind("<Button-1>", open_github)

        # ----------------------------
        # Main Content: Left (Controls) & Right (Log)
        # ----------------------------
        content_frame = ttk.Frame(root, padding=(10, 10))
        content_frame.pack(fill="both", expand=True)
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=2)

        # Left Panel: Controls
        left_frame = ttk.Frame(content_frame, padding=(10, 10))
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        # File Selection Group
        file_frame = ttk.LabelFrame(left_frame, text="File Selection", padding=(10, 10))
        file_frame.pack(fill="x", pady=5)
        self.load_button = ttk.Button(file_frame, text="Load CSV", command=self.load_csv)
        self.load_button.pack(side="left")
        self.file_label = ttk.Label(file_frame, text="No file selected", foreground="gray")
        self.file_label.pack(side="left", padx=10)

        # Model Selection Group
        model_frame = ttk.LabelFrame(left_frame, text="Model Selection", padding=(10, 10))
        model_frame.pack(fill="x", pady=5)
        self.stg_radio = ttk.Radiobutton(model_frame, text="STG model", variable=self.selected_model, value="STG")
        self.stg_radio.pack(side="left", padx=5)
        self.da_radio = ttk.Radiobutton(model_frame, text="DA model", variable=self.selected_model, value="DA")
        self.da_radio.pack(side="left", padx=5)

        # CPU Selection Group
        cpu_frame = ttk.LabelFrame(left_frame, text="CPU Selection", padding=(10, 10))
        cpu_frame.pack(fill="x", pady=5)
        cpu_label = ttk.Label(cpu_frame, text="Select number of CPUs:")
        cpu_label.pack(side="left")
        self.cpu_selector = ttk.Combobox(cpu_frame,
                                         textvariable=self.selected_cpus,
                                         values=list(range(1, self.num_cpus + 1)),
                                         state="readonly",
                                         width=5)
        self.cpu_selector.pack(side="left", padx=5)
        self.cpu_selector.current(self.selected_cpus.get() - 1)

        # Python Environment Selection Group
        env_frame = ttk.LabelFrame(left_frame, text="Python Environment", padding=(10, 10))
        env_frame.pack(fill="x", pady=5)
        self.env_button = ttk.Button(env_frame, text="Select Python Env", command=self.select_python_env)
        self.env_button.pack(side="left")
        self.env_label = ttk.Label(env_frame, text=os.path.basename(self.python_env), foreground="gray")
        self.env_label.pack(side="left", padx=10)
        self.save_config_button = ttk.Button(env_frame, text="Save Default Env", command=self.save_config)
        self.save_config_button.pack(side="left", padx=5)

        # GPU Options Group
        gpu_frame = ttk.LabelFrame(left_frame, text="GPU Options", padding=(10, 10))
        gpu_frame.pack(fill="x", pady=5)
        self.use_gpu = tk.BooleanVar(value=False)
        self.gpu_checkbox = ttk.Checkbutton(gpu_frame, text="Use GPU", variable=self.use_gpu)
        self.gpu_checkbox.pack(side="left")

        # Pipeline Actions Group
        action_frame = ttk.Frame(left_frame, padding=(10, 10))
        action_frame.pack(fill="x", pady=5)
        self.run_button = ttk.Button(action_frame, text="Use the pipeline", command=self.run_pipeline, state=tk.DISABLED)
        self.run_button.pack(side="top", fill="x", pady=2)
        self.kill_button = ttk.Button(action_frame, text="Kill Process", command=self.kill_process, state=tk.DISABLED)
        self.kill_button.pack(side="top", fill="x", pady=2)
        self.save_button = ttk.Button(action_frame, text="Save Results", command=self.save_results, state=tk.DISABLED)
        self.save_button.pack(side="top", fill="x", pady=2)

        # Help Button
        help_button = ttk.Button(left_frame, text="Help", command=self.show_help)
        help_button.pack(pady=5, anchor="center")

        # Right Panel: Log Output
        right_frame = ttk.Frame(content_frame, padding=(10, 10))
        right_frame.grid(row=0, column=1, sticky="nsew")
        log_label = ttk.Label(right_frame, text="Log Output:")
        log_label.pack(anchor="w")
        self.log_area = scrolledtext.ScrolledText(right_frame, wrap="word", width=50, height=20,
                                                  state=tk.DISABLED, font=("Courier", 10))
        self.log_area.pack(fill="both", expand=True, pady=5)

        # Drag & Drop support for CSV files
        self.root.drop_target_register(tkdnd.DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.drop_csv)

        # Load default Python environment if config.json exists and update GPU checkbox.
        self.load_config()
        self.update_gpu_checkbox()

    def load_config(self):
        """Loads the default python environment from config.json if it exists."""
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base_dir = os.getcwd()
        config_path = os.path.join(base_dir, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config_data = json.load(f)
                if "python_env" in config_data:
                    self.python_env = config_data["python_env"]
                    if hasattr(self, "env_label"):
                        self.env_label.config(text=os.path.basename(self.python_env))
                    self.update_log(f"Loaded default Python environment from config: {self.python_env}")
            except Exception as e:
                self.update_log(f"Error loading config: {str(e)}")

    def check_gpu_availability(self):
        """
        Returns True if the selected python environment can import torch
        and torch.cuda.is_available() returns True, otherwise False.
        """
        try:
            result = subprocess.run(
                [self.python_env, "-c", "import torch; print(torch.cuda.is_available())"],
                capture_output=True, text=True, timeout=10
            )
            return result.stdout.strip() == "True"
        except Exception:
            return False

    def update_gpu_checkbox(self):
        """Update the GPU checkbox state and value based on the availability of CUDA."""
        available = self.check_gpu_availability()
        if available:
            self.use_gpu.set(True)
            self.gpu_checkbox.config(state=tk.NORMAL)
            self.update_log("CUDA is available. 'Use GPU' enabled and checked.")
        else:
            self.use_gpu.set(False)
            self.gpu_checkbox.config(state=tk.DISABLED)
            self.update_log("CUDA not available or PyTorch not installed. 'Use GPU' disabled.")

    def show_help(self):
        help_text = (
            "Instructions:\n"
            "1. Load a CSV file using the 'Load CSV' button or drag and drop it onto the window.\n"
            "2. Select a model (STG or DA).\n"
            "3. Choose the number of CPUs.\n"
            "4. (Optional) Select a Python environment that has the required dependencies.\n"
            "5. If PyTorch with CUDA is available in the selected environment, you can choose to use GPU.\n"
            "6. Click 'Use the pipeline' to start processing.\n"
            "7. You can kill the process if needed and save the results once done.\n"
            "8. Use 'Save Default Env' to store the current Python environment in config.json."
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
            # Re-check GPU availability when a new interpreter is chosen.
            self.update_gpu_checkbox()

    def save_config(self):
        """Saves the current Python environment path to config.json in the app folder."""
        config_data = {"python_env": self.python_env}
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
        self.save_button.config(state=tk.DISABLED)  # disable until signal received
        thread = threading.Thread(target=self.execute_script, daemon=True)
        thread.start()

    def execute_script(self):
        python_exe = self.python_env or sys.executable
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.abspath(".")
        script_path = os.path.join(base_path, "script.py")
        # Pass the GPU flag (as "True"/"False") as an extra parameter.
        command = [
            python_exe,
            script_path,
            self.csv_file,
            self.selected_model.get(),
            str(self.selected_cpus.get()),
            self.result_file,
            str(self.use_gpu.get())
        ]
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        signal_received = False
        for line in self.process.stdout:
            stripped_line = line.strip()
            # Hide the special signal from the log.
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
        save_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                 filetypes=[("CSV Files", "*.csv")])
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
