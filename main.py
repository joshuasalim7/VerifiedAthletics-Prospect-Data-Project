import os
import pandas as pd
import numpy as np
import re

# Global mapping of standard column names to possible variants.
COLUMN_GROUPS = {
    'player_name': ['name', 'name_more_info_click_name', 'player', 'athlete', 'student_name', 'full_name'],
    'jersey_number': ['#', 'number', 'jersey', 'jersey_#', 'no'],
    'position': ['pos', 'position1', 'position2', 'primary_position'],
    'height': ['ht', 'height_', 'player_height'],
    'weight': ['wt', 'wt_lb', 'weight_', 'player_weight'],
    'weighted_gpa': ['weighted_gpa', 'wgpa', 'gpa_weighted'],
    'core_gpa': ['core_gpa', 'cgpa', 'gpa_core'],
    'twitter_handle': ['twitter', 'twitter_handle', 'x_handle'],
    'hudl_link': ['hudl', 'hudl_profile', 'highlights'],
    'offers': ['offers', 'college_offers', 'scholarships'],
    'year': ['grad_year', 'graduation_year', 'class', 'class_of'],
    'phone': ['phone', 'cell', 'mobile', 'contact']
}

# List of columns to get info (except school_name, which we add later).
REQUIRED_COLUMNS = ['player_name', 'year', 'position', 'jersey_number', 'height',
                      'weight', 'weighted_gpa', 'core_gpa', 'twitter_handle',
                      'hudl_link', 'offers', 'phone']

def extract_school_name(file_path):
    file_name = os.path.basename(file_path)
    unwanted_terms = ["football", "prospect", "recruiting", "list", "sheet", "updated"]
    for term in unwanted_terms:
        file_name = file_name.lower().replace(term, "")
    file_name = ''.join([char for char in file_name if not char.isdigit()])
    school_name = file_name.replace(".csv", "").replace(".xlsx", "").replace(".xls", "").strip()
    school_name = ' '.join([word.capitalize() for word in school_name.split()])
    return school_name

def load_and_clean_file(file_path):
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, header=None, dtype=str)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path, header=None, dtype=str)
        else:
            raise ValueError("Unsupported file format. Please use CSV or Excel files.")

        df.columns = [f"col_{i}" for i in range(df.shape[1])]

        df = df.dropna(how='all').reset_index(drop=True)

        df['school_name'] = extract_school_name(file_path)

        return df
    except Exception as e:
        return f"Error processing {file_path}: {e}"

def combine_prospect_files(file_paths, output_file):
    cleaned_dataframes = []

    for file_path in file_paths:
        df = load_and_clean_file(file_path)
        if isinstance(df, pd.DataFrame):
            cleaned_dataframes.append(df)
        else:
            print(df)

    if cleaned_dataframes:
        final_df = pd.concat(cleaned_dataframes, ignore_index=True)
        final_df.to_csv(output_file, index=False)
        print(f"Combined data saved to {output_file}")
    else:
        print("No valid data to combine.")

if __name__ == "__main__":
    file_paths = [
        '/content/West Orange Football Prospect Sheet.csv',
        '/content/Ingram Prospect Sheet.xlsx',
        '/content/Central 2024 Prospect Sheet Updated.xlsx',
        '/content/Stevens PROSPECT LIST.xlsx',
        '/content/Hammond Recruiting - Updated  - Form Responses 1.csv'
    ]
    output_file = 'combined_prospect_sheets.csv'

    if not file_paths:
        print("No prospect sheet files found.")
    else:
        combine_prospect_files(file_paths, output_file)
