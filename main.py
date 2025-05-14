import yaml
import pandas as pd
import io
import uuid # For unique IDs for processing steps
import json # For parsing user_mappings_json
import re # For pattern matching in content analysis

# --- Added imports for explicit type checking and typing ---
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype, is_string_dtype 
from typing import Dict, List, Any # Added List, Any, Dict
# --- End Added imports ---

from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles # If we add CSS later

from thefuzz import fuzz # Import thefuzz

CONFIG_FILE_PATH = 'rules.yaml'
DEFAULT_FUZZY_THRESHOLD = 85 # Define default threshold
TEMPLATE_MATCH_THRESHOLD = 75 # Minimum % of template's defined input columns that must match
FUZZY_HEADER_MATCH_THRESHOLD = 85 # Threshold for fuzzy matching headers within a template

app = FastAPI()

# Mount static files (e.g., for CSS, JS) - not used yet but good practice
# app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# In-memory storage for uploaded file content and mapping results (simple approach)
# For production, consider a more robust temporary storage or cache
processing_steps = {} 

# --- Content Analysis Helpers ---

def looks_like_date(series: pd.Series, sample_size=50, threshold=0.8) -> bool:
    """Check if a high threshold of non-null values in a series can be parsed as dates."""
    
    # Handle empty or all-null series
    if series.isnull().all():
        return False
        
    # If already datetime objects, it's definitely date-like
    if is_datetime64_any_dtype(series):
        return True

    # If it's numeric or generic object, try converting the whole series
    # We do this *before* string check because sometimes dates are stored as numbers (Excel dates)
    # or objects that pandas might recognize directly.
    if is_numeric_dtype(series) or series.dtype == 'object':
        try:
            # Attempt conversion on the whole series (dropna might help performance/avoid errors)
            pd.to_datetime(series.dropna(), errors='raise') 
            # If the above worked without error for the non-null values, assume it's date-like
            return True 
        except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime):
            # Conversion failed for numeric/object. If it's *purely* numeric and failed, 
            # it's unlikely to be date strings later. If object, it *might* contain strings.
            if is_numeric_dtype(series):
                 return False # Pure numeric that didn't convert? Not dates.
            # If it was object, it might still contain strings that look like dates,
            # so fall through to the string/sampling check below.
            pass 
        except Exception as e:
            # Catch other unexpected errors during conversion
            print(f"Warning: Unexpected error during initial date conversion attempt for series '{series.name}': {e}")
            # Fall through to sampling as a best effort.
            pass

    # Now, specifically handle string types or object types that might contain strings
    if is_string_dtype(series) or series.dtype == 'object':
        # Use sampling for potentially string-formatted dates
        sampled = series.dropna().sample(min(sample_size, len(series.dropna())))
        if sampled.empty:
            return False
        
        try:
            # Attempt conversion on the sample
            converted = pd.to_datetime(sampled, errors='coerce')
            # Check the ratio of successful conversions
            success_ratio = converted.notna().sum() / len(sampled)
            return success_ratio >= threshold
        except Exception as e: # Catch any unexpected error during sample conversion
            print(f"Warning: Unexpected error during sampled date conversion for series '{series.name}': {e}")
            return False
            
    # If not datetime, numeric, object, or string after checks, it's likely not a date column
    return False

def looks_like_ssn(series: pd.Series, sample_size=50, threshold=0.8) -> bool:
    """Check if a high threshold of values match SSN patterns (XXX-XX-XXXX or XXXXXXXXX)."""
    if series.isnull().all(): return False
    sampled = series.dropna().astype(str).sample(min(sample_size, len(series.dropna())))
    if sampled.empty: return False
    
    # Regex for SSN: optional hyphens
    ssn_pattern = re.compile(r'^\d{3}-?\d{2}-?\d{4}$') 
    match_count = sampled.apply(lambda x: bool(ssn_pattern.match(x.strip()))).sum()
    return (match_count / len(sampled)) >= threshold

def looks_like_zip(series: pd.Series, sample_size=50, threshold=0.8) -> bool:
    """Check if a high threshold of values match ZIP code patterns (XXXXX or XXXXX-XXXX)."""
    if series.isnull().all(): return False
    sampled = series.dropna().astype(str).sample(min(sample_size, len(series.dropna())))
    if sampled.empty: return False
    
    zip_pattern = re.compile(r'^\d{5}(-\d{4})?$')
    match_count = sampled.apply(lambda x: bool(zip_pattern.match(x.strip()))).sum()
    return (match_count / len(sampled)) >= threshold

def looks_like_state_abbr(series: pd.Series, sample_size=50, threshold=0.8) -> bool:
    """Check if a high threshold of values look like 2-letter state abbreviations."""
    if series.isnull().all(): return False
    sampled = series.dropna().astype(str).sample(min(sample_size, len(series.dropna())))
    if sampled.empty: return False
    
    # Simple check for 2 uppercase letters
    state_pattern = re.compile(r'^[A-Z]{2}$')
    match_count = sampled.apply(lambda x: bool(state_pattern.match(x.strip()))).sum()
    # TODO: Could add check against actual list of state abbreviations for higher accuracy
    return (match_count / len(sampled)) >= threshold

def has_consistent_delimiter(series: pd.Series, delimiter=' ', sample_size=50, threshold=0.7) -> bool:
    """Check if a delimiter consistently appears in non-null string values."""
    if series.isnull().all() or not pd.api.types.is_string_dtype(series):
        return False
    sampled = series.dropna().astype(str).sample(min(sample_size, len(series.dropna())))
    if sampled.empty: return False
    
    contains_delimiter_count = sampled.apply(lambda x: delimiter in x).sum()
    return (contains_delimiter_count / len(sampled)) >= threshold

# --- End Content Analysis Helpers ---

def load_rules(config_path):
    """Loads the configuration, including standard columns and aliases."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Successfully loaded rules from {config_path}")
        return {
            'standard_output_columns': config.get('standard_output_columns', []),
            'column_name_aliases': config.get('column_name_aliases', {}),
            'transformation_rules': config.get('transformation_rules', []),
            'mapping_templates': config.get('mapping_templates', {})
        }
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at {config_path}")
        return {'standard_output_columns': [], 'column_name_aliases': {}, 'transformation_rules': [], 'mapping_templates': {}}
    except yaml.YAMLError as e:
        print(f"ERROR: Could not parse YAML configuration at {config_path}: {e}")
        return {'standard_output_columns': [], 'column_name_aliases': {}, 'transformation_rules': [], 'mapping_templates': {}}

# --- Enhanced Normalization Function ---
# Define common prefixes/suffixes to strip (lowercase)
COMMON_PREFIXES_SUFFIXES = [
    'member', 'subscriber', 'dependent', 'dep', 
    'employee', 'emp', 
    'col', 'column', 
    '#', 'no', 'num', 'number'
]

def normalize_column_name(name):
    """Converts column name to lowercase, removes common prefixes/suffixes, replaces separators, and strips whitespace."""
    if not isinstance(name, str):
        return name # Return original if not a string

    normalized = name.lower().strip() # Initial lowercase and strip

    # Remove common prefixes/suffixes (whole word match for safety)
    # We iterate multiple times in case of stacked prefixes (e.g., "member_dep_fname")
    # This is simple; more robust might use regex word boundaries.
    changed = True
    while changed:
        changed = False
        for prefix in COMMON_PREFIXES_SUFFIXES:
            prefix_space = prefix + ' '
            if normalized.startswith(prefix_space):
                normalized = normalized[len(prefix_space):].strip()
                changed = True
        for suffix in COMMON_PREFIXES_SUFFIXES:
             space_suffix = ' ' + suffix
             if normalized.endswith(space_suffix):
                normalized = normalized[:-len(space_suffix)].strip()
                changed = True
        # Also handle cases without space, e.g., MemberID -> ID (less safe)
        # Let's comment this part out for now, as it might be too aggressive
        # for prefix in COMMON_PREFIXES_SUFFIXES:
        #     if normalized.startswith(prefix):
        #        # Check if the next char is not alpha (e.g. MemberID vs Memberly) - crude
        #        if len(normalized) > len(prefix) and not normalized[len(prefix)].isalpha():
        #             normalized = normalized[len(prefix):].strip()
        #             changed = True
        # for suffix in COMMON_PREFIXES_SUFFIXES:
        #     if normalized.endswith(suffix):
        #         # Check if prev char is not alpha
        #         if len(normalized) > len(suffix) and not normalized[-len(suffix)-1].isalpha():
        #             normalized = normalized[:-len(suffix)].strip()
        #             changed = True

    # Replace hyphens and underscores with spaces AFTER stripping prefixes/suffixes
    normalized = normalized.replace('-', ' ').replace('_', ' ')
    
    # Final clean up: collapse multiple spaces and strip again
    normalized = ' '.join(normalized.split())
    
    return normalized.strip() # Final strip just in case
# --- End Enhanced Normalization Function ---

# --- Restore missing function definition ---
# Modified to accept file_content as bytes
def load_input_data_from_bytes(file_content_bytes: bytes, filename: str):
    """Loads the input CSV data from bytes using pandas."""
    try:
        # Use io.BytesIO to treat the byte string as a file
        df = pd.read_csv(io.BytesIO(file_content_bytes))
        print(f"Successfully loaded input data from uploaded file: {filename}")
        return df
    except pd.errors.EmptyDataError:
        print(f"ERROR: Uploaded CSV file is empty: {filename}")
        return None
    except Exception as e:
        print(f"ERROR: Could not load input data from {filename}: {e}")
        return None
# --- End restore missing function definition ---

def map_columns(input_column_names, standard_columns_config, column_name_aliases, fuzzy_match_threshold=85):
    """
    Compares input columns to standard defined columns using various strategies
    and reports on the mapping.
    Strategies: Exact match, Normalized match (case-insensitive, stripped), Alias match, Fuzzy match.
    """
    mapped_to_standard = {}  # Stores: standard_name -> {'input': original_input_col, 'method': '...'}
    used_input_cols = set()

    standard_col_definitions = {col['name']: col for col in standard_columns_config}
    
    input_col_normalized_map = {normalize_column_name(c): c for c in input_column_names}

    # Pass 1: Exact Matches
    for std_name in standard_col_definitions.keys():
        if std_name in input_column_names and std_name not in used_input_cols:
            mapped_to_standard[std_name] = {'input': std_name, 'method': 'exact'}
            used_input_cols.add(std_name)

    # Pass 2: Normalized Matches
    for std_name in standard_col_definitions.keys():
        if std_name in mapped_to_standard: continue
        norm_std_name = normalize_column_name(std_name)
        if norm_std_name in input_col_normalized_map:
            original_input_name = input_col_normalized_map[norm_std_name]
            if original_input_name not in used_input_cols:
                mapped_to_standard[std_name] = {'input': original_input_name, 'method': 'normalized'}
                used_input_cols.add(original_input_name)

    # Pass 3: Alias Matches
    aliases_for_standard = {}
    for alias_key, std_val in column_name_aliases.items():
        if std_val not in aliases_for_standard:
            aliases_for_standard[std_val] = set()
        aliases_for_standard[std_val].add(alias_key)

    for std_name in standard_col_definitions.keys():
        if std_name in mapped_to_standard: continue
        if std_name in aliases_for_standard:
            for alias_key in aliases_for_standard[std_name]:
                if alias_key in input_column_names and alias_key not in used_input_cols:
                    mapped_to_standard[std_name] = {'input': alias_key, 'method': f'alias (exact: {alias_key})'}
                    used_input_cols.add(alias_key)
                    break 
                norm_alias_key = normalize_column_name(alias_key)
                if norm_alias_key in input_col_normalized_map:
                    original_input_for_norm_alias = input_col_normalized_map[norm_alias_key]
                    if original_input_for_norm_alias not in used_input_cols:
                        mapped_to_standard[std_name] = {'input': original_input_for_norm_alias, 'method': f'alias (normalized: {alias_key} -> {original_input_for_norm_alias})'}
                        used_input_cols.add(original_input_for_norm_alias)
                        break
            if std_name in mapped_to_standard: continue

    # Pass 4: Fuzzy Matches (for remaining standard columns and remaining input columns)
    remaining_std_names = [s_name for s_name in standard_col_definitions.keys() if s_name not in mapped_to_standard]
    remaining_input_cols = [ic_name for ic_name in input_column_names if ic_name not in used_input_cols]

    for std_name in remaining_std_names:
        best_match_input_col = None
        highest_score = 0
        best_match_method = '' # Keep track of which ratio was best

        norm_std_name = normalize_column_name(std_name)

        for input_col_name in remaining_input_cols:
            norm_input_col = normalize_column_name(input_col_name)
            if not norm_std_name or not norm_input_col: continue # Skip empty normalized names
            
            # Calculate both token_set_ratio and partial_ratio
            score_tsr = fuzz.token_set_ratio(norm_std_name, norm_input_col)
            score_pr = fuzz.partial_ratio(norm_std_name, norm_input_col)
            
            # Use the maximum of the two scores
            score = max(score_tsr, score_pr)
            current_method = f'fuzzy (tsr={score_tsr}, pr={score_pr} -> max={score}%)' # More detailed method string
            
            if score > highest_score and score >= fuzzy_match_threshold:
                highest_score = score
                best_match_input_col = input_col_name
                best_match_method = current_method # Store the method details
        
        if best_match_input_col:
            # --- Existing logic to check if this input col is better for another std col --- 
            # (This logic might need refinement now that we use max score)
            # It primarily compares based on the final 'highest_score'
            is_better_for_other = False
            for mapped_std, details in mapped_to_standard.items():
                if details['method'].startswith('fuzzy') and details['input'] == best_match_input_col:
                    try:
                        # Extract max score from the method string (e.g., "... max=88%)")
                        existing_score = int(re.search(r'max=(\d+)%', details['method']).group(1))
                        if existing_score >= highest_score:
                            is_better_for_other = True
                            break
                    except (AttributeError, ValueError, TypeError):
                         print(f"Warning: Could not parse existing fuzzy score from '{details['method']}'")
                         # If we can't parse, assume it might be better, play safe
                         if mapped_to_standard[mapped_std].get('_score', 0) >= highest_score: # Check if we stored a raw score before
                              is_better_for_other = True
                              break
            
            if not is_better_for_other and best_match_input_col not in used_input_cols:
                 can_use = True
                 # Check if any other std_name could have claimed this input_col with a higher max score
                 norm_best_match_input_col = normalize_column_name(best_match_input_col)
                 if norm_best_match_input_col: # Ensure normalized name is not empty
                     for other_std_name in remaining_std_names:
                         if other_std_name == std_name: continue
                         norm_other_std = normalize_column_name(other_std_name)
                         if not norm_other_std: continue

                         other_score_tsr = fuzz.token_set_ratio(norm_other_std, norm_best_match_input_col)
                         other_score_pr = fuzz.partial_ratio(norm_other_std, norm_best_match_input_col)
                         other_score = max(other_score_tsr, other_score_pr)

                         if other_score > highest_score and other_score >= fuzzy_match_threshold:
                             can_use = False
                             break
                 if can_use:
                    mapped_to_standard[std_name] = {
                        'input': best_match_input_col, 
                        'method': best_match_method, # Use the detailed method string
                        '_score': highest_score # Store raw score for potential future comparisons
                    }
                    used_input_cols.add(best_match_input_col) # Mark as used

    missing_standard_columns = sorted([s_name for s_name in standard_col_definitions.keys() if s_name not in mapped_to_standard])
    unmapped_input_columns = sorted([c_name for c_name in input_column_names if c_name not in used_input_cols])

    return {
        "mapped_columns": mapped_to_standard,
        "missing_standard": missing_standard_columns,
        "unmapped_input": unmapped_input_columns,
    }

def parse_address(address_str, address_type='city_state'):
    """
    Parse address strings into their components.
    address_type can be:
    - 'city_state': Handles "City, State" or "City State" format
    - 'city_state_zip': Handles "City, State ZIP" or "City State ZIP" format
    """
    if not isinstance(address_str, str):
        return None, None, None
    
    address_str = address_str.strip()
    if not address_str:
        return None, None, None
    
    city = None
    state = None
    zip_code = None
    
    # Try to split by comma first (most common format)
    parts = [p.strip() for p in address_str.split(',')]
    
    if len(parts) >= 2:
        city = parts[0]
        # Handle state and zip
        state_zip = parts[1].strip()
        # Try to split state and zip
        state_zip_parts = state_zip.split()
        if len(state_zip_parts) >= 2:
            state = state_zip_parts[0]
            zip_code = ' '.join(state_zip_parts[1:])
        else:
            state = state_zip_parts[0]
    else:
        # No comma, try to split by spaces
        parts = address_str.split()
        if len(parts) >= 2:
            # Assume last part is state, everything before is city
            state = parts[-1]
            city = ' '.join(parts[:-1])
            # If we have more parts, the last two might be state and zip
            if len(parts) >= 3:
                state = parts[-2]
                zip_code = parts[-1]
                city = ' '.join(parts[:-2])
    
    return city, state, zip_code

def apply_split_rule(rule, df_input, existing_transformed_cols):
    rule_name = rule.get('name', rule.get('input_column', 'UI_Rule'))
    transformation_type = rule.get('transformation_type') or rule.get('type')
    
    if transformation_type == 'split_name_by_delimiter':
        input_col_name = rule.get('input_column')
        options = rule.get('options', {})
        delimiter = options.get('delimiter', ' ') 
        output_map = rule.get('output_mapping', {})

        if not input_col_name or not output_map:
            print(f"Warning: Transformation rule '{rule_name}' is missing 'input_column' or 'output_mapping'. Skipping.")
            return

        if input_col_name in df_input.columns:
            series_to_split = df_input[input_col_name].astype(str).fillna('')
            max_part_needed = 0
            try:
                max_part_needed = max(int(p.split('_')[1]) for p in output_map.values() if p and p.startswith('part_'))
            except (ValueError, AttributeError):
                print(f"Warning: Invalid part key in output_mapping for rule '{rule_name}'. Example: 'part_1'. Skipping rule.")
                return
            
            n_splits = max(0, max_part_needed - 1)
            parts_df = series_to_split.str.split(delimiter, n=n_splits, expand=True)
            for col_idx in parts_df.columns: # Strip whitespace
                parts_df[col_idx] = parts_df[col_idx].str.strip()

            for std_col_target, part_key in output_map.items():
                if std_col_target in existing_transformed_cols: # Already processed by a higher priority rule
                    continue
                if part_key and part_key.startswith('part_'):
                    try:
                        part_index = int(part_key.split('_')[1]) - 1
                        if 0 <= part_index < parts_df.shape[1]:
                            existing_transformed_cols[std_col_target] = parts_df[part_index]
                        else:
                            existing_transformed_cols[std_col_target] = pd.Series([pd.NA] * len(df_input), index=df_input.index)
                    except (ValueError, AttributeError):
                         print(f"Warning: Invalid part key '{part_key}' for rule '{rule_name}'. Skipping this mapping for {std_col_target}.")
        else:
            print(f"Warning: Input column '{input_col_name}' for transformation rule '{rule_name}' not found. Skipping.")
    
    elif transformation_type == 'parse_address':
        input_col_name = rule.get('input_column')
        address_type = rule.get('options', {}).get('address_type', 'city_state')
        output_map = rule.get('output_mapping', {})
        
        if not input_col_name or not output_map:
            print(f"Warning: Address transformation rule '{rule_name}' is missing 'input_column' or 'output_mapping'. Skipping.")
            return
            
        if input_col_name in df_input.columns:
            # Apply address parsing to the entire series
            parsed_addresses = df_input[input_col_name].apply(lambda x: parse_address(x, address_type))
            
            # Create a DataFrame from the parsed addresses
            parsed_df = pd.DataFrame(parsed_addresses.tolist(), 
                                   columns=['city', 'state', 'zip_code'],
                                   index=df_input.index)
            
            # Map the parsed components to standard columns
            for std_col_target, component in output_map.items():
                if std_col_target in existing_transformed_cols:  # Skip if already processed
                    continue
                if component in ['city', 'state', 'zip_code']:
                    existing_transformed_cols[std_col_target] = parsed_df[component]
        else:
            print(f"Warning: Input column '{input_col_name}' for address transformation rule '{rule_name}' not found. Skipping.")

def analyze_and_suggest(input_df, mapping_results, standard_columns_config):
    """
    Analyzes content of unmapped columns and suggests mappings or transformations.
    Modifies mapping_results in-place with new suggestions.
    Returns a list of suggested transformation rules.
    """
    print("Starting content analysis for suggestions...")
    suggested_transformations = []
    unmapped_input_cols = mapping_results.get('unmapped_input', [])
    mapped_to_standard = mapping_results.get('mapped_columns', {})
    used_input_cols = set(details['input'] for details in mapped_to_standard.values())
    standard_col_map = {col['name']: col for col in standard_columns_config}
    available_standard_cols = [s_name for s_name in standard_col_map.keys() if s_name not in mapped_to_standard]

    if not unmapped_input_cols:
        print("No unmapped input columns to analyze.")
        return suggested_transformations

    print(f"Analyzing {len(unmapped_input_cols)} unmapped columns: {unmapped_input_cols}")

    for input_col_name in unmapped_input_cols:
        if input_col_name not in input_df.columns:
            continue # Should not happen, but safety check
        
        series = input_df[input_col_name]
        if series.isnull().all():
             print(f"  Skipping '{input_col_name}': all null values.")
             continue

        analyzed = False # Flag to track if we made a suggestion for this column

        # 1. Suggest based on content type match with available standard columns
        for std_col_name in available_standard_cols:
            std_col_info = standard_col_map[std_col_name]
            std_type = std_col_info.get('type', 'string')
            suggestion_made = False

            if std_type == 'date' and looks_like_date(series):
                mapped_to_standard[std_col_name] = {'input': input_col_name, 'method': 'content (date-like)'}
                suggestion_made = True
            elif std_col_name == 'SSN' and looks_like_ssn(series):
                mapped_to_standard[std_col_name] = {'input': input_col_name, 'method': 'content (SSN-like)'}
                suggestion_made = True
            elif std_col_name == 'ZipCode' and looks_like_zip(series):
                mapped_to_standard[std_col_name] = {'input': input_col_name, 'method': 'content (ZIP-like)'}
                suggestion_made = True
            elif std_col_name == 'State' and looks_like_state_abbr(series):
                 mapped_to_standard[std_col_name] = {'input': input_col_name, 'method': 'content (State Abbr-like)'}
                 suggestion_made = True
            # Add more type/pattern checks here (e.g., phone, gender)

            if suggestion_made:
                print(f"  Suggested mapping by content: '{input_col_name}' -> '{std_col_name}' ({mapped_to_standard[std_col_name]['method']})")
                used_input_cols.add(input_col_name)
                available_standard_cols.remove(std_col_name)
                analyzed = True
                break # Move to next input column once a suggestion is made
        
        if analyzed: continue # Already suggested a direct map for this input col

        # 2. Suggest transformations if no direct map was suggested
        # Suggest splitting Full Name
        if has_consistent_delimiter(series, delimiter=' ') and \
           'FirstName' in available_standard_cols and \
           'LastName' in available_standard_cols:
            suggested_transformations.append({
                "type": "split_name_by_delimiter",
                "input_column": input_col_name,
                "options": { "delimiter": " " },
                "output_mapping": { "FirstName": "part_1", "LastName": "part_2" },
                "suggestion_reason": f"Column '{input_col_name}' contains spaces; potential First/Last Name."
            })
            print(f"  Suggested transformation: Split '{input_col_name}' for First/Last Name.")
            # Don't mark input col as used here, as transformation creates new outputs
            analyzed = True

        # Suggest parsing City, State (or City State)
        elif (has_consistent_delimiter(series, delimiter=',') or has_consistent_delimiter(series, delimiter=' ')) and \
             'City' in available_standard_cols and \
             'State' in available_standard_cols:
             # Basic check: does the last part look like a state?
             # This is very heuristic!
             last_part_looks_statey = False
             try:
                 last_parts = series.dropna().astype(str).apply(lambda x: x.split(',')[-1].strip().split()[-1])
                 if looks_like_state_abbr(last_parts, threshold=0.6): # Lower threshold for this guess
                     last_part_looks_statey = True
             except Exception:
                 pass # Ignore errors in this heuristic check

             if last_part_looks_statey:
                suggested_transformations.append({
                    "type": "parse_address",
                    "input_column": input_col_name,
                    "options": { "address_type": "city_state_zip" }, # Guess most common, user can change
                    "output_mapping": { "City": "city", "State": "state", "ZipCode": "zip_code" if "ZipCode" in available_standard_cols else None },
                    "suggestion_reason": f"Column '{input_col_name}' contains separators and ends with state-like values; potential City/State/Zip."
                })
                print(f"  Suggested transformation: Parse '{input_col_name}' for Address parts.")
                analyzed = True
        
        # Add more transformation suggestions here (e.g., different delimiters, other address formats)

    # Update mapping_results with any new direct mappings found by content
    mapping_results['mapped_columns'] = mapped_to_standard
    # Recalculate unmapped/missing based on suggestions
    final_used_input = set(details['input'] for details in mapped_to_standard.values())
    mapping_results['unmapped_input'] = sorted([c_name for c_name in input_df.columns if c_name not in final_used_input])
    mapping_results['missing_standard'] = sorted([s_name for s_name in standard_col_map.keys() if s_name not in mapped_to_standard])
    
    print("Content analysis finished.")
    return suggested_transformations

# --- FastAPI Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "mapping_results": None, "output_df_html": None, "step_id": None})

@app.post("/uploadfile/", response_class=JSONResponse)
async def create_upload_file(
    request: Request, 
    file: UploadFile = File(...)
):
    if not file.filename.endswith('.csv'):
        return JSONResponse(status_code=400, content={"error": "Invalid file type. Please upload a CSV."})

    contents_bytes = await file.read() # Read file content as bytes
    
    rules_data = load_rules(CONFIG_FILE_PATH)
    standard_columns_config = rules_data.get('standard_output_columns')
    column_aliases = rules_data.get('column_name_aliases')
    mapping_templates = rules_data.get('mapping_templates', {}) # Load mapping templates

    standard_column_names_only = [col['name'] for col in standard_columns_config] if standard_columns_config else []

    # Use the default threshold for the initial mapping
    current_fuzzy_threshold = DEFAULT_FUZZY_THRESHOLD
    print(f"Performing initial mapping with default fuzzy threshold: {current_fuzzy_threshold}%")

    if not standard_columns_config:
        return JSONResponse(status_code=500, content={"error": "Could not load standard column definitions."})

    input_df = load_input_data_from_bytes(contents_bytes, file.filename)

    if input_df is None:
        return JSONResponse(status_code=400, content={"error": f"Could not process CSV file: {file.filename}"})

    all_input_column_names = list(input_df.columns) # Get all input column names
    normalized_input_names_set = {normalize_column_name(col) for col in all_input_column_names}

    # --- File-level Template Matching Logic ---
    best_template_match = None
    highest_template_score = 0
    matched_template_name = None
    template_derived_mappings = {}

    if mapping_templates:
        print(f"Found {len(mapping_templates)} mapping templates. Attempting file-level match...")
        for template_name, template_data in mapping_templates.items():
            # This is the {StandardColumnKeyInRules: ExpectedInputColumnNameInFile}
            template_column_map_config = template_data.get('mapping', {})
            if not template_column_map_config: 
                print(f"  Skipping template '{template_name}': No mapping definitions.")
                continue

            # These are the *input column names* the template expects to find in an uploaded file.
            # We need to normalize them for comparison.
            expected_template_input_names = {normalize_column_name(val) for val in template_column_map_config.values() if val}
            if not expected_template_input_names:
                print(f"  Skipping template '{template_name}': No input column names defined in its mapping values.")
                continue
            
            match_count = 0
            # Compare against the normalized column names from the *uploaded file*
            for norm_expected_input_name in expected_template_input_names:
                if norm_expected_input_name in normalized_input_names_set:
                    match_count += 1
            
            # Score is based on how many of the template's expected input columns were found in the uploaded file.
            score = (match_count / len(expected_template_input_names)) * 100
            print(f"  Template '{template_name}': {match_count}/{len(expected_template_input_names)} of its defined input columns matched uploaded file. Score: {score:.2f}%")

            if score > highest_template_score:
                highest_template_score = score
                best_template_match = template_data # Store the whole template data
                matched_template_name = template_name
    
    initial_mapping_method = "column_by_column"
    if best_template_match and highest_template_score >= TEMPLATE_MATCH_THRESHOLD:
        print(f"Good file-level match found: Template '{matched_template_name}' with score {highest_template_score:.2f}%")
        initial_mapping_method = f"template_match ({matched_template_name})"
        
        # Construct `mapped_columns` based on the matched template
        # Template `mapping` is {StandardCol: InputColFromTemplateDef}
        # We need to find the *actual* input col from the uploaded file that matches InputColFromTemplateDef
        final_template_map = {}
        template_map_config = best_template_match.get('mapping', {})

        # Create a reverse map from normalized input from file to original input from file
        input_norm_to_orig_map = {normalize_column_name(orig_name): orig_name for orig_name in all_input_column_names}

        for std_col_target, template_input_col_def in template_map_config.items():
            norm_template_input_def = normalize_column_name(template_input_col_def)
            if norm_template_input_def in input_norm_to_orig_map:
                actual_input_col_name = input_norm_to_orig_map[norm_template_input_def]
                final_template_map[std_col_target] = {
                    'input': actual_input_col_name,
                    'method': f'template ({matched_template_name}: {template_input_col_def} -> {actual_input_col_name})'
                }
            # Else: The input column defined in template wasn't found in the uploaded file (even if overall score was high)
        
        mapping_results = {
            "mapped_columns": final_template_map,
            "missing_standard": [], # Will be populated by analyze_and_suggest
            "unmapped_input": [],   # Will be populated by analyze_and_suggest
        }
        # Update used_input_cols based on template mapping to prepare for analyze_and_suggest
        used_input_cols_from_template = {val['input'] for val in final_template_map.values()}
        remaining_input_for_analysis = [col for col in all_input_column_names if col not in used_input_cols_from_template]
        
        # We still run analyze_and_suggest to pick up anything the template missed or content-based items.
        # It will operate on the `mapping_results` (which now contains template mappings)
        # and the remaining input columns.
        # We need to pass a modified input_df or list of columns for analyze_and_suggest to operate on remaining.
        # For simplicity, let analyze_and_suggest work on the full df but be aware of existing template mappings.
        # analyze_and_suggest modifies mapping_results in-place and re-calculates missing/unmapped. 
        print("Running content analysis after applying template...")
        suggested_transformations = analyze_and_suggest(input_df, mapping_results, standard_columns_config)

    else:
        if mapping_templates: # Only print if templates were loaded
            print(f"No suitable file-level template match found (highest score: {highest_template_score:.2f}% for '{matched_template_name}', threshold: {TEMPLATE_MATCH_THRESHOLD}%). Proceeding with column-by-column mapping.")
            matched_template_name = None # Ensure it's None if threshold not met
        
        # Fallback to original column-by-column mapping if no good template match
        mapping_results = map_columns(all_input_column_names, standard_columns_config, column_aliases, fuzzy_match_threshold=current_fuzzy_threshold)
        suggested_transformations = analyze_and_suggest(input_df, mapping_results, standard_columns_config)

    step_id = str(uuid.uuid4())
    processing_steps[step_id] = {
        "filename": file.filename,
        "rules_data": rules_data,
        "input_df_bytes": contents_bytes, # Store raw bytes to recreate DF later
        "all_input_columns": all_input_column_names, # Store for efficiency
        "mapping_results": mapping_results, # Store initial results
        "fuzzy_threshold_used": current_fuzzy_threshold, # Store initial threshold
        "suggested_transformations": suggested_transformations, # Store initial suggestions
        "matched_template_name": matched_template_name, # Send to frontend
        "initial_mapping_method": initial_mapping_method, # Send how initial map was derived
    }
    
    # Generate preview of the input DataFrame
    input_df_preview_html = "<p>No data to preview.</p>"
    if not input_df.empty:
        try:
            # classes='table table-striped table-hover' can be added for bootstrap-like styling if using bootstrap
            input_df_preview_html = input_df.head().to_html(max_rows=5, classes=['dataframe', 'preview-table'], justify='left', border=0)
        except Exception as e:
            print(f"Error generating HTML preview for input_df: {e}")
            input_df_preview_html = "<p>Error generating data preview.</p>"

    # Return the initial mapping results
    return JSONResponse(content={
        "step_id": step_id,
        "filename": file.filename,
        "input_preview_html": input_df_preview_html,
        "all_input_columns": all_input_column_names, 
        "standard_column_names": standard_column_names_only,
        "matched_template_name": matched_template_name, # Send to frontend
        "initial_mapping_method": initial_mapping_method, # Send how initial map was derived
        "mapping_results": mapping_results,
        "initial_fuzzy_threshold": current_fuzzy_threshold, # Send the initial threshold to UI
        "suggested_transformations": suggested_transformations
    })

@app.post("/remap/", response_class=JSONResponse)
async def remap_columns_endpoint(
    request: Request, 
    step_id: str = Form(...),
    new_threshold: int = Form(...)
):
    """ Endpoint to re-calculate column mappings with a new fuzzy threshold. """
    if step_id not in processing_steps:
        return JSONResponse(status_code=404, content={"error": "Process ID not found or expired. Please upload again."})

    if not (0 <= new_threshold <= 100):
         return JSONResponse(status_code=400, content={"error": "Invalid fuzzy threshold. Must be between 0 and 100."})

    print(f"Remapping step {step_id} with new threshold: {new_threshold}%")
    
    stored_data = processing_steps[step_id]
    input_df_bytes = stored_data.get("input_df_bytes")
    rules_data = stored_data.get("rules_data")
    all_input_columns = stored_data.get("all_input_columns")
    filename = stored_data.get("filename", "unknown_file") # Use filename for logging

    if not all([input_df_bytes, rules_data, all_input_columns]):
        return JSONResponse(status_code=500, content={"error": "Incomplete data for remapping step. Please try again."})

    standard_columns_config = rules_data.get('standard_output_columns')
    column_aliases = rules_data.get('column_name_aliases')

    # Re-run mapping with the new threshold
    new_mapping_results = map_columns(all_input_columns, standard_columns_config, column_aliases, fuzzy_match_threshold=new_threshold)

    # Re-run content analysis based on the *new* mapping results
    # Reload the DataFrame for analysis
    input_df_for_analysis = load_input_data_from_bytes(input_df_bytes, filename)
    if input_df_for_analysis is None:
         return JSONResponse(status_code=500, content={"error": "Could not reload data for content analysis during remap."})
         
    new_suggested_transformations = analyze_and_suggest(input_df_for_analysis, new_mapping_results, standard_columns_config)

    # Update stored data (optional, but keeps state consistent if user proceeds)
    stored_data["mapping_results"] = new_mapping_results
    stored_data["suggested_transformations"] = new_suggested_transformations
    stored_data["fuzzy_threshold_used"] = new_threshold 
    
    # Return the updated mapping results and suggestions
    return JSONResponse(content={
        "step_id": step_id, # Keep the same step_id
        "mapping_results": new_mapping_results,
        "suggested_transformations": new_suggested_transformations,
        "fuzzy_threshold_used": new_threshold
    })

@app.post("/processfile/", response_class=JSONResponse)
async def process_file_endpoint(
    request: Request, 
    step_id: str = Form(...),
    user_mappings_json: str = Form(...) # New parameter for user's choices
):
    if step_id not in processing_steps:
        return JSONResponse(status_code=404, content={"error": "Process ID not found or expired. Please upload again."})

    stored_data = processing_steps[step_id]
    input_df_bytes = stored_data.get("input_df_bytes")
    rules_data = stored_data.get("rules_data")
    filename = stored_data.get("filename", "uploaded_file.csv")

    # Parse the user_mappings_json
    user_map_data = {}
    ui_transformations = []
    direct_column_maps = {}
    try:
        user_map_data = json.loads(user_mappings_json)
        # Expecting a structure like: {
        #   "ui_transformations": [ { "type": "split_name_by_delimiter", "input_column": ..., "options": ..., "output_mapping": ... } ],
        #   "direct_column_maps": { "StandardCol": "InputCol", ... }
        # }
        ui_transformations = user_map_data.get("ui_transformations", [])
        direct_column_maps = user_map_data.get("direct_column_maps", {})

    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "Invalid user mapping data submitted."})

    if not all([input_df_bytes, rules_data]):
        return JSONResponse(status_code=500, content={"error": "Incomplete data for processing step. Please try again."})

    standard_columns_config = rules_data.get('standard_output_columns')
    loaded_transformation_rules = rules_data.get('transformation_rules', [])
    
    input_df = load_input_data_from_bytes(input_df_bytes, filename) # Recreate DataFrame
    if input_df is None:
        return JSONResponse(status_code=500, content={"error": "Could not reload data for processing."})

    standard_col_names_for_output = [col['name'] for col in standard_columns_config]
    # Initialize standardized_df with the input_df's index to ensure proper alignment
    standardized_df = pd.DataFrame(index=input_df.index, columns=standard_col_names_for_output)

    transformed_columns_data = {} # Stores {std_col_name: pd.Series_of_data}

    # --- Apply Transformations (Priority: UI-defined, then rules.yaml) ---
    
    # 1. Apply UI-defined transformations
    print(f"Applying {len(ui_transformations)} UI-defined transformations.")
    for ui_rule in ui_transformations:
        apply_split_rule(ui_rule, input_df, transformed_columns_data)

    # 2. Apply transformations from rules.yaml (only if output column not already filled)
    print(f"Applying {len(loaded_transformation_rules)} transformations from rules.yaml.")
    for yaml_rule in loaded_transformation_rules:
        apply_split_rule(yaml_rule, input_df, transformed_columns_data)
    
    # Populate the standardized DataFrame from transformations
    for std_col_name, data_series in transformed_columns_data.items():
        if std_col_name in standardized_df.columns:
            standardized_df[std_col_name] = data_series

    # Priority 2: From user-defined direct mappings (for columns not handled by transformations)
    for std_col_name, mapped_input_col_name in direct_column_maps.items():
        if std_col_name in transformed_columns_data: # Already populated by a transformation
            continue
        
        if mapped_input_col_name and mapped_input_col_name != '__UNMAPPED__':
            if mapped_input_col_name in input_df.columns:
                standardized_df[std_col_name] = input_df[mapped_input_col_name]
            else:
                print(f"Warning: User-mapped input column '{mapped_input_col_name}' for standard '{std_col_name}' not found in DataFrame. Column will be NaN.")
                # No need to assign pd.NA, as DataFrame is initialized with NaNs
        # else: Column explicitly unmapped by user, or not mapped, remains NaN
    
    # Ensure all standard columns exist in the standardized_df, even if not in user_mappings
    missing_standard_columns = sorted([s_name for s_name in standard_col_names_for_output if s_name not in standardized_df.columns])
    if missing_standard_columns:
        print(f"Warning: The following standard columns are missing in the standardized DataFrame: {', '.join(missing_standard_columns)}")

    # Convert standardized_df to HTML
    output_df_html = "<p>No data processed or empty result.</p>"
    if not standardized_df.empty:
        try:
            output_df_html = standardized_df.to_html(max_rows=20, classes=['dataframe', 'output-table'], justify='left', border=0, na_rep='-')
        except Exception as e:
            print(f"Error generating HTML for output_df: {e}")
            output_df_html = "<p>Error generating output data table.</p>"
    
    # Optionally, clear the step from memory after processing to save space
    # del processing_steps[step_id]

    return JSONResponse(content={"output_df_html": output_df_html})

# --- Function to safely write rules back to YAML ---
# NOTE: This basic version overwrites the file. 
# Consider backups or more robust merging if needed.
# Using ruamel.yaml could preserve comments/formatting better if installed.
def save_rules(config_path: str, rules_data: Dict[str, Any]) -> bool:
    """Saves the provided rules data dictionary back to the YAML file."""
    try:
        with open(config_path, 'w') as f:
            # Use options for better readability if possible
            yaml.dump(rules_data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
        print(f"Successfully saved updated rules to {config_path}")
        return True
    except IOError as e:
        print(f"ERROR: Could not write to configuration file {config_path}: {e}")
        return False
    except yaml.YAMLError as e:
        print(f"ERROR: Could not dump YAML data for saving: {e}")
        return False
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while saving rules: {e}")
        return False
# --- End save_rules function ---

# --- Add endpoint to save a new mapping template --- 
@app.post("/savetemplate/", response_class=JSONResponse)
async def save_template_endpoint(
    request: Request,
    step_id: str = Form(...), # To get original columns if needed
    template_name: str = Form(...),
    template_description: str = Form(""), # Optional description
    direct_mappings_json: str = Form(...) # JSON string of {StandardCol: InputCol}
):
    """ Saves the current user-defined direct column mappings as a new template in rules.yaml. """
    print(f"Received request to save template: '{template_name}'")

    if not template_name:
        return JSONResponse(status_code=400, content={"error": "Template name cannot be empty."})

    # --- Get required info from stored step data --- 
    if step_id not in processing_steps:
         return JSONResponse(status_code=404, content={"error": "Process ID not found or expired. Cannot retrieve original columns."})    
    stored_data = processing_steps[step_id]
    original_input_columns = stored_data.get("all_input_columns")
    if not original_input_columns:
         # This shouldn't happen if step_id is valid, but safety check
         return JSONResponse(status_code=500, content={"error": "Could not retrieve original input columns for the template."})    

    # --- Parse the submitted mappings --- 
    try:
        direct_mappings = json.loads(direct_mappings_json)
        if not isinstance(direct_mappings, dict):
             raise ValueError("Mappings must be a dictionary.")
        # Basic validation: ensure values are strings or null (representing unmapped)
        # We only save the mapped ones into the template
        template_mapping = {k: v for k, v in direct_mappings.items() if v and isinstance(v, str)}
        if not template_mapping: # Don't save empty templates
            return JSONResponse(status_code=400, content={"error": "Cannot save template with no columns mapped."})    
            
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON format for direct mappings."})
    except ValueError as e:
         return JSONResponse(status_code=400, content={"error": f"Invalid mapping data: {e}"})

    # --- Load current rules --- 
    rules_data = load_rules(CONFIG_FILE_PATH)
    # Check if load_rules returned defaults due to an error
    # This is tricky, as load_rules prints errors but returns defaults.
    # We might add a check here, or rely on the fact that save_rules might fail if the structure is bad.

    # --- Check for existing template name --- 
    if template_name in rules_data.get('mapping_templates', {}):
        # For now, reject duplicates. Could add overwrite option later.
        return JSONResponse(status_code=400, content={"error": f"Template name '{template_name}' already exists."})

    # --- Add the new template data --- 
    if 'mapping_templates' not in rules_data:
        rules_data['mapping_templates'] = {}
        
    rules_data['mapping_templates'][template_name] = {
        'description': template_description,
        'original_input_columns': original_input_columns, # Store for reference
        'mapping': template_mapping # Store the cleaned {Standard: Input} map
    }

    # --- Save the updated rules back to the file --- 
    save_successful = save_rules(CONFIG_FILE_PATH, rules_data)

    if save_successful:
        return JSONResponse(content={
            "message": f"Template '{template_name}' saved successfully.",
            "template_name": template_name
        })
    else:
        # save_rules already printed an error
        return JSONResponse(status_code=500, content={"error": "Failed to save the updated rules file. Check server logs."})
