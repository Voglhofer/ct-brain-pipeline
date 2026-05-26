import os

csv_root = os.environ.get('SEQUENCE_CSV_ROOT', os.path.join(os.getcwd(), 'csv'))
feature_path = os.environ.get('SEQUENCE_FEATURE_ROOT', os.path.join(os.getcwd(), 'features'))
final_output_path = os.environ.get('SEQUENCE_FINAL_OUTPUT_ROOT', os.path.join(os.getcwd(), 'FinalSubmission'))