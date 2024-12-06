import os
import pandas as pd
import numpy as np
import trimesh
from tqdm import tqdm

def determine_final_level(row):
    try:
        # Choose the person who labeled the level if one value is N/A
        if pd.isna(row['Inside level (Person 1)']) and not pd.isna(row['Inside level (Person 2)']):
            return int(row['Inside level (Person 2)'])
        elif pd.isna(row['Inside level (Person 2)']) and not pd.isna(row['Inside level (Person 1)']):
            return int(row['Inside level (Person 1)'])

        # If both persons provided a level, choose based on confidence
        if row['Inside level confident (Person 1)'] > row['Inside level confident (Person 2)']:
            return int(row['Inside level (Person 1)'])
        elif row['Inside level confident (Person 2)'] > row['Inside level confident (Person 1)']:
            return int(row['Inside level (Person 2)'])

        # If both confidences are equal and available
        if row['Inside level confident (Person 1)'] == 1 and row['Inside level confident (Person 2)'] == 1:
            return round((int(row['Inside level (Person 1)']) + int(row['Inside level (Person 2)'])) / 2)
    except (ValueError, TypeError):
        return None

    return int(row['Inside level (Person 1)'])

def load_and_validate_obj_file(file_path):
    """
    Load and validate an OBJ file to check for integrity.
    Returns vertices if the file is valid, otherwise returns None.
    Also checks if the file exists for Inside level classification.
    """
    if not os.path.exists(file_path):
        return None, f"File not found: {file_path}"
    
    try:
        mesh = trimesh.load(file_path)
        # Check if the loaded object is a Scene or a Mesh
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) == 0:
                return None, f"Skipping file {file_path}: No geometries found in Scene."
            # Extract the first geometry
            mesh = list(mesh.geometry.values())[0]
        
        vertices = mesh.vertices
        if len(vertices) == 0:
            return None, f"Skipping file {file_path}: No vertices found."
        return vertices, None
    except Exception as e:
        return None, f"Error loading file {file_path}: {e}"

def process_obj_files(base_dir, data, augment=False):
    """
    Process all OBJ files, validate them, and apply augmentations if specified.
    """
    validated_labels = []
    error_messages = []

    for index, row in tqdm(data.iterrows(), total=data.shape[0], desc="Processing OBJ files", leave=True):
        layer_folder = str(row['Object ID (Dataset Original Object ID)']).strip()
        obj_file_path = os.path.join(base_dir, layer_folder, f"{layer_folder}.obj")

        vertices, error_message = load_and_validate_obj_file(obj_file_path)
        if vertices is not None:
            validated_labels.append(row)
        else:
            if error_message:
                error_messages.append(error_message)

    # Print all error messages at the end to avoid interfering with the progress bar
    for message in error_messages:
        tqdm.write(message)

    # Create a new DataFrame with validated labels only
    validated_labels_df = pd.DataFrame(validated_labels)
    validated_labels_df.reset_index(drop=True, inplace=True)
    validated_labels_df['Count No.'] = validated_labels_df.index + 1
    return validated_labels_df

def check_inside_level_classification(base_dir, excel_path):
    # Load the Excel file to get information about objects and their inside levels
    try:
        data = pd.read_excel(excel_path)
    except FileNotFoundError:
        print(f"Error: The file '{excel_path}' does not exist.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

    # Apply the function to create the 'Final Inside Level' column
    data['Final Inside Level'] = data.apply(determine_final_level, axis=1)

    # Remove rows with NaN values in 'Final Inside Level'
    data = data.dropna(subset=['Final Inside Level'])

    # Convert 'Final Inside Level' to integer and remove rows where the level is 0
    data['Final Inside Level'] = data['Final Inside Level'].astype(int)
    data = data[data['Final Inside Level'] > 0]

    # Process OBJ files
    validated_labels_df = process_obj_files(base_dir, data)

    # Save the validated labels to the final Excel file
    output_file_path = 'Dataset/hermanmiller/label/Final_Validated_Regularity_Levels.xlsx'
    validated_labels_df.to_excel(output_file_path, index=False)
    print(f"Validated data saved to {output_file_path}")

    # Summary
    total_files = data.shape[0]
    validated_count = len(validated_labels_df)
    failed_count = total_files - validated_count
    print("\nSummary:")
    print(f"Total files processed: {total_files}")
    print(f"Successfully validated files: {validated_count}")
    print(f"Failed validations: {failed_count}")

    return validated_labels_df

# Example usage
if __name__ == "__main__":
    base_dir = "Dataset\hermanmiller\obj-hermanmiller"
    excel_path = 'Dataset\hermanmiller\label\HermanMiller.xlsx'
    check_inside_level_classification(base_dir, excel_path)