import pandas as pd
import os

def add_disease_to_dataset(disease_name, description, symptoms, dataset_path):
    print(f"DEBUG: Adding disease to dataset CSV: {disease_name}, {description}, {symptoms}, {dataset_path}")
    # Load dataset
    df = pd.read_csv(dataset_path)
    # Prepare new row
    new_row = {col: '' for col in df.columns}
    new_row['Disease'] = disease_name
    # Fill symptoms columns
    symptoms_list = [s.strip() for s in symptoms.split(',') if s.strip()]
    print(f"DEBUG: Parsed symptoms list: {symptoms_list}")
    for i, symptom in enumerate(symptoms_list):
        col_name = f'Symptom_{i+1}'
        if col_name in new_row:
            new_row[col_name] = symptom
    # Optionally add description if column exists
    if 'Description' in new_row:
        new_row['Description'] = description
    print(f"DEBUG: New row to add: {new_row}")
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(dataset_path, index=False)
    return True
