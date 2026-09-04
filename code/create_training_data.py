#import sys
#from subprocess import Popen
import subprocess
import os

dataset_dir = '../dataset'
regions_json = f"{dataset_dir}/regions.json"     # strings only =(  )

training_data_dir = "dataset_training"
eval_data_dir = "dataset_evaluation"

#Popen("mkdir -p dataset_training", shell=True).wait()
os.makedirs(training_data_dir, exist_ok=True)

#Popen("mkdir dataset_evaluation", shell=True).wait()
os.makedirs(eval_data_dir, exist_ok=True)


#args = [regions_json, dataset_dir, training_data_dir]

#Popen("python3 hdmapeditor/create_dataset_for_training.py ../dataset/regions.json ../dataset/ dataset_training/", shell=True).wait()
r = subprocess.run(["python3", "hdmapeditor/create_dataset_for_training.py", regions_json, dataset_dir, training_data_dir], capture_output=True, text=True)

print(f"stdout: {r.stdout}")
print(f"stderr: {r.stderr}")

#Popen("python3 hdmapeditor/create_dataset_for_training_vectors.py ../dataset/regions.json ../dataset/ dataset_training/", shell=True).wait()

