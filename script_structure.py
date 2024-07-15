import os
import json

def get_project_structure(root_dir):
    project_structure = {
        "root": root_dir,
        "directories": [],
        "files": [],
        "requirements": [],
        "main_file": None,
        "index_file": None
    }
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            project_structure["directories"].append(os.path.relpath(os.path.join(dirpath, dirname), root_dir))
        for filename in filenames:
            filepath = os.path.relpath(os.path.join(dirpath, filename), root_dir)
            project_structure["files"].append(filepath)
            if filename == "requirements.txt":
                with open(os.path.join(dirpath, filename), 'r') as f:
                    project_structure["requirements"] = f.read().splitlines()
            elif filename == "main.py":
                project_structure["main_file"] = filepath
            elif filename == "index.html":
                project_structure["index_file"] = filepath

    return project_structure

def write_to_json(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    root_directory = "."  # Répertoire courant du projet
    output_json_file = "project_structure.json"

    project_data = get_project_structure(root_directory)
    write_to_json(project_data, output_json_file)

    print(f"Project structure has been written to {output_json_file}")
