import kagglehub

# Download latest version
path = kagglehub.dataset_download("insaff/massachusetts-roads-dataset")

print("Path to dataset files:", path)