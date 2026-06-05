import os

# Configuration
BACKEND_DIR = "Backend"
ML_MODEL_DIR = "ml_model"
BACKEND_OUT = "backend_core.txt"
ML_OUT = "ml_core.txt"

EXCLUDED_DIRS = {"node_modules", "__pycache__", ".git", "venv", "env"}
EXCLUDED_EXTS = {".jpg", ".jpeg", ".png", ".mp4", ".csv", ".json", ".h5", ".tflite", ".pkl", ".pt", ".weights"}
INCLUDE_BACKEND_EXT = {".js"}
INCLUDE_ML_EXT = {".py", ".ipynb"}

def should_exclude(path):
    parts = set(path.replace("\\", "/").split("/"))
    return not parts.isdisjoint(EXCLUDED_DIRS)

def aggregate_files(root_dir, out_file, include_exts):
    with open(out_file, "w", encoding="utf-8") as out:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Remove excluded directories from traversal
            dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in include_exts:
                    continue
                if ext in EXCLUDED_EXTS:
                    continue
                fpath = os.path.join(dirpath, fname)
                if should_exclude(fpath):
                    continue
                rel_path = os.path.relpath(fpath, start=os.getcwd())
                # Write header
                out.write("// ==========================================\n")
                out.write(f"// FILE: {rel_path.replace(os.sep, '/')}\n")
                out.write("// ==========================================\n\n")
                try:
                    with open(fpath, "r", encoding="utf-8") as src:
                        out.write(src.read())
                except Exception as e:
                    out.write(f"// [Could not read file: {e}]\n")
                out.write("\n\n")

if __name__ == "__main__":
    aggregate_files(BACKEND_DIR, BACKEND_OUT, INCLUDE_BACKEND_EXT)
    aggregate_files(ML_MODEL_DIR, ML_OUT, INCLUDE_ML_EXT)
    print("Aggregation complete.")