import subprocess, shutil, torch
print("torch.cuda.is_available():", torch.cuda.is_available())
nvsmi = shutil.which("nvidia-smi")
if nvsmi:
    print("nvidia-smi found at", nvsmi)
    try:
        out = subprocess.check_output([nvsmi], stderr=subprocess.STDOUT).decode("utf-8", errors="ignore")
        print(out[:1000])
    except Exception as e:
        print("Failed to run nvidia-smi:", e)
else:
    print("nvidia-smi not found on PATH")
