import kagglehub

# Download latest version
path = kagglehub.dataset_download("vermakeshav/churn-datacsv")

print("Path to dataset files:", path)