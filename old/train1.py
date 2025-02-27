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

# List of required columns (except school_name, which we add later).
REQUIRED_COLUMNS = ['player_name', 'year', 'position', 'jersey_number', 'height',
                      'weight', 'weighted_gpa', 'core_gpa', 'twitter_handle',
                      'hudl_link', 'offers', 'phone']

def extract_school_name(file_path):
    file_name = file_path.split('/')[-1].split('\\')[-1]
    unwanted_terms = ["football", "prospect", "recruiting", "list", "sheet", "updated"]
    for term in unwanted_terms:
        file_name = file_name.lower().replace(term, "")
    file_name = ''.join([char for char in file_name if not char.isdigit()])
    school_name = file_name.replace(".csv", "").replace(".xlsx", "").replace(".xls", "").strip()
    school_name = ' '.join([word.capitalize() for word in school_name.split()])
    return school_name

def standardize_column_names(df):
    # Clean column names: strip, lower, replace spaces, and remove punctuation.
    df.columns = (df.columns.astype(str)
                  .str.strip()
                  .str.lower()
                  .str.replace(' ', '_')
                  .str.replace('[^\w\s]', '', regex=True))

    # Loop through each standard column defined in COLUMN_GROUPS.
    for standard_name, variations in COLUMN_GROUPS.items():
        # Find columns whose name exactly matches any variant or contains one.
        matching_cols = [col for col in df.columns if any(var == col or var in col for var in variations)]
        if not matching_cols:
            # Try a more lenient matching if an exact match wasn't found.
            matching_cols = [col for col in df.columns if any(var in col for var in variations)]
        if matching_cols:
            if len(matching_cols) > 1:
                # Use backfill across columns (first non-null value) if multiple candidates exist.
                df[standard_name] = df[matching_cols].bfill(axis=1).iloc[:, 0]
                df = df.drop(columns=matching_cols)
            else:
                df = df.rename(columns={matching_cols[0]: standard_name})
    return df

def process_year_groups(df):
    # Prepare a "year" column that will be filled based on detecting 20xx patterns.
    df['year'] = None
    current_year = None

    # Identify header rows by checking for common header keywords.
    header_mask = df.apply(
        lambda row: row.astype(str)
                    .str.contains('Name|Position|HT|WT', case=False, na=False)
                    .any(),
        axis=1
    )

    # Identify rows that contain a year (e.g. 2025 or Class of 2025).
    year_mask = df.apply(
        lambda row: row.astype(str).str.contains(r'\b20\d{2}\b', case=False, na=False).any(),
        axis=1
    )

    rows_to_drop = []
    for idx, row in df.iterrows():
        if year_mask.iloc[idx]:
            # Get all cells that contain a year and update the current_year variable.
            year_cells = row[row.astype(str).str.contains(r'\b20\d{2}\b', case=False, na=False)]
            if not year_cells.empty:
                cell_str = str(year_cells.iloc[0])
                found = re.findall(r'\b20\d{2}\b', cell_str)
                if found:
                    current_year = int(found[0])
                    rows_to_drop.append(idx)
        elif header_mask.iloc[idx]:
            rows_to_drop.append(idx)
        else:
            # Assign the current_year to this row.
            df.at[idx, 'year'] = current_year

    # Remove rows that are either headers or year markers.
    df = df.drop(rows_to_drop).reset_index(drop=True)
    return df

def clean_height_weight(df):
    if 'height' in df.columns:
        # Remove extraneous characters from height and ensure a proper format.
        df['height'] = df['height'].astype(str).str.replace('"', '').str.replace('′', "'")
        df['height'] = df['height'].apply(lambda x: x if "'" in str(x) else (f"{int(float(x))}'0\""
                                                                            if x.replace('.', '').isdigit()
                                                                            else x))
    if 'weight' in df.columns:
        # Extract numeric values from the weight field.
        df['weight'] = pd.to_numeric(df['weight'].astype(str).str.extract('(\d+\.?\d*)')[0], errors='coerce')
    return df

def remove_invalid_rows(df):
    # Remove rows with text patterns that indicate an invalid or header row.
    invalid_patterns = {
        'player_name': ["Qr Code", "*CLICK ON", "nan", "CLASS OF"],
        'jersey_number': ["*CLICK ON", "Qr Code", "nan"]
    }
    def is_invalid(row):
        for col, patterns in invalid_patterns.items():
            if col in row.index and any(pattern.lower() in str(row[col]).lower() for pattern in patterns):
                return True
        return False
    df = df[~df.apply(is_invalid, axis=1)]
    return df

def load_and_clean_file(file_path):
    try:
        # Load the file according to its extension.
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, header=None, dtype=str)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, header=None, dtype=str)
        else:
            raise ValueError("Unsupported file format. Please use CSV or Excel files.")

        # Temporarily set generic column names.
        df.columns = [f"col_{i}" for i in range(df.shape[1])]

        # ----------- Robust Header Detection -----------
        header_row_index = None
        max_match_count = 0
        # Loop through each row to find the one with the most header-like cells.
        for i, row in df.iterrows():
            match_count = 0
            for cell in row:
                if isinstance(cell, str):
                    cell_lower = cell.lower()
                    # Count a match if any variant from any standard column is found.
                    for variants in COLUMN_GROUPS.values():
                        if any(var in cell_lower for var in variants):
                            match_count += 1
            if match_count > max_match_count:
                max_match_count = match_count
                header_row_index = i

        # --- Fallback to first row as header if no header-like row was detected ---
        if header_row_index is None or max_match_count == 0:
            header_row_index = 0

        # Use the identified (or fallback) header row.
        df.columns = df.iloc[header_row_index].fillna('').astype(str)
        df = df[header_row_index + 1:].reset_index(drop=True)

        # Standardize the column names.
        df = standardize_column_names(df)
        # Process rows that contain year information.
        df = process_year_groups(df)
        # Clean height and weight columns.
        df = clean_height_weight(df)
        # Remove rows that are clearly invalid.
        df = remove_invalid_rows(df)
        # Drop completely empty rows.
        df = df.dropna(how='all').reset_index(drop=True)

        # Ensure all required columns exist.
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan

        # If duplicate column names exist, differentiate them.
        df.columns = pd.Index([f"{col}_{i}" if list(df.columns).count(col) > 1 else col
                               for i, col in enumerate(df.columns)])

        return df
    except Exception as e:
        return f"Error processing {file_path}: {e}"

def combine_prospect_files(file_paths, output_file):
    cleaned_dataframes = []

    for file_path in file_paths:
        try:
            df = load_and_clean_file(file_path)
            if isinstance(df, pd.DataFrame):
                # Add the school name extracted from the file path.
                df['school_name'] = extract_school_name(file_path)
                # --- Loop through each required column and do some basic validation ---
                for col in ['player_name', 'year', 'position', 'jersey_number', 'height', 'weight',
                            'weighted_gpa', 'core_gpa', 'twitter_handle', 'hudl_link', 'offers', 'phone']:
                    # If a column is missing, add it as NaN.
                    if col not in df.columns:
                        df[col] = np.nan
                    else:
                        # For numeric columns, attempt to convert their values.
                        if col in ['jersey_number', 'weight', 'weighted_gpa', 'core_gpa']:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        # For the year column, check that it looks like a valid graduation year.
                        if col == 'year':
                            df[col] = df[col].apply(lambda x: int(x) if str(x).isdigit() and 2000 < int(x) < 2100 else np.nan)
                cleaned_dataframes.append(df)
            else:
                print(df)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if cleaned_dataframes:
        final_df = pd.concat(cleaned_dataframes, ignore_index=True)
        final_df = final_df.dropna(how='all')
        # Remove any leftover header-like rows.
        final_df = final_df[~final_df.apply(lambda row:
            row.astype(str).str.contains('Name|Position|HT|WT', case=False, na=False).any(), axis=1)]

        # Reorder the columns: standard columns first, then any additional ones.
        standard_columns = ['player_name', 'school_name', 'year', 'position', 'jersey_number',
                            'height', 'weight', 'weighted_gpa', 'core_gpa', 'twitter_handle',
                            'hudl_link', 'offers', 'phone']
        existing_standard_cols = [col for col in standard_columns if col in final_df.columns]
        other_cols = [col for col in final_df.columns if col not in standard_columns]
        final_df = final_df[existing_standard_cols + other_cols]

        final_df.to_csv(output_file, index=False)
        print(f"Combined data saved to {output_file}")
    else:
        print("No valid data to combine.")

input_files = [
    '/content/West Orange Football Prospect Sheet.csv',
    '/content/Ingram Prospect Sheet.xlsx',
    '/content/Central 2024 Prospect Sheet Updated.xlsx',
    '/content/Stevens PROSPECT LIST.xlsx',
    '/content/Hammond Recruiting - Updated  - Form Responses 1.csv'
]

output_file = '14prospect_sheets.csv'
combine_prospect_files(input_files, output_file)
