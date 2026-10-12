import subprocess

def run_script(script_name):
    print(f"Running {script_name}...")
    subprocess.run(["python", script_name], check=True)
    print(f"{script_name} completed.")

if __name__ == "__main__":
    run_script("data_preprocessing.py")
    print("\n \n --------------------------- \n \n")
    run_script("model_training.py")
    print("\n \n --------------------------- \n \n")
    run_script("model_evaluation.py")
    print("\n \n --------------------------- \n \n")
    print("All scripts executed.")
