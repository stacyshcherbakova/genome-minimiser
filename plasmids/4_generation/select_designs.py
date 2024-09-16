import os
import csv
from Bio import SeqIO

def clean_label(label):
    """Remove descriptors like 'promoter' from labels to enable matching."""
    return label.lower().replace(" promoter", "").replace("(fragment)", "").strip()

def analyze_gbk_file_for_rep_origin_and_features(gbk_file):
    """Analyze a GenBank file to find promoters, CDSs with >=95% identity, and origin of replication with 100% identity."""
    promoters = set()
    cdss = set()
    rep_origin_found = False
    rep_origin_identity = 0

    try:
        for record in SeqIO.parse(gbk_file, "genbank"):
            print(f"Parsing {record.id} in file {gbk_file}...")
            for feature in record.features:
                if feature.type == "CDS":
                    identity = float(feature.qualifiers.get('identity', [0])[0])
                    if identity >= 95:
                        label = feature.qualifiers.get('label', [""])[0]
                        cleaned_label = clean_label(label)
                        print(f"  Found CDS with label: {label} (cleaned: {cleaned_label}) and identity {identity}%")
                        cdss.add(cleaned_label)
                elif feature.type == "promoter":
                    identity = float(feature.qualifiers.get('identity', [0])[0])
                    if identity >= 95:
                        label = feature.qualifiers.get('label', [""])[0]
                        cleaned_label = clean_label(label)
                        print(f"  Found promoter with label: {label} (cleaned: {cleaned_label}) and identity {identity}%")
                        promoters.add(cleaned_label)
                elif feature.type == "rep_origin":
                    identity = float(feature.qualifiers.get('identity', [0])[0])
                    if identity == 100:
                        rep_origin_found = True
                        rep_origin_identity = identity
                        print(f"  Found origin of replication with identity {identity}%: {feature.qualifiers.get('label', ['Unknown'])[0]}")

        print(f"CDS Labels: {cdss}")
        print(f"Promoter Labels: {promoters}")
        print(f"Origin of Replication Found: {rep_origin_found} with Identity: {rep_origin_identity}%")

    except Exception as e:
        print(f"Error parsing {gbk_file}: {e}")
    
    return promoters, cdss, rep_origin_found, rep_origin_identity

def process_files_in_directory(directory):
    """Process all .gbk files in the given directory and find promoters, CDSs, and origin of replication."""
    summary = []

    for file_name in os.listdir(directory):
        if file_name.endswith(".gbk"):
            file_path = os.path.join(directory, file_name)
            print(f"Processing file: {file_name}")
            promoters, cdss, rep_origin_found, rep_origin_identity = analyze_gbk_file_for_rep_origin_and_features(file_path)
            if promoters or cdss or rep_origin_found:
                summary.append({
                    'file_name': file_name,
                    'promoters': ', '.join(promoters),
                    'cdss': ', '.join(cdss),
                    'rep_origin_found': rep_origin_found,
                    'rep_origin_identity': rep_origin_identity
                })

    return summary

def save_summary_to_csv(summary, output_csv):
    """Save the collected data to a CSV file."""
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['file_name', 'promoters', 'cdss', 'rep_origin_found', 'rep_origin_identity'])
        writer.writeheader()
        for entry in summary:
            writer.writerow(entry)

# Directory containing the GenBank files
directory = "/Users/anastasiiashcherbakova/Desktop/5_generation_plannotate/annotations"
# Output CSV file
output_csv = "plasmid_summary.csv"

# Process the files and save the results to CSV
summary = process_files_in_directory(directory)
save_summary_to_csv(summary, output_csv)

print(f"Results saved to {output_csv}")

# Output a count of plasmids with 100% replication origin and/or promoters/CDSs found
print("********************************************************")
print("Plasmids with promoters, CDSs, or origin of replication")
print("********************************************************")
count = sum(1 for entry in summary if entry['rep_origin_found'])
print(f"Total plasmids with 100% origin of replication identity: {count}")
count2 = sum(1 for entry in summary if entry['rep_origin_found'] and entry['promoters'] and entry['cdss'])
print(f"Total plasmids with origin of replication, CDS, and promoter: {count2}")


# import os
# from Bio import SeqIO

# def clean_label(label):
#     """Remove descriptors like 'promoter' from labels to enable matching."""
#     return label.lower().replace(" promoter", "").replace("(fragment)", "").strip()

# def analyze_gbk_file_for_matching_names_and_rep_origin(gbk_file):
#     """Analyze a GenBank file to find all matching promoter and CDS names and check for origin of replication with >95% identity."""
#     cds_labels = set()
#     promoter_labels = set()
#     rep_origin_found = False
#     rep_origin_identity = 0

#     try:
#         for record in SeqIO.parse(gbk_file, "genbank"):
#             print(f"Parsing {record.id} in file {gbk_file}...")
#             for feature in record.features:
#                 if feature.type == "CDS":
#                     identity = float(feature.qualifiers.get('identity', [0])[0])
#                     if identity >= 95:
#                         label = feature.qualifiers.get('label', [""])[0]
#                         cleaned_label = clean_label(label)
#                         print(f"  Found CDS with label: {label} (cleaned: {cleaned_label}) and identity {identity}%")
#                         cds_labels.add(cleaned_label)
#                 elif feature.type == "promoter":
#                     label = feature.qualifiers.get('label', [""])[0]
#                     cleaned_label = clean_label(label)
#                     print(f"  Found promoter with label: {label} (cleaned: {cleaned_label})")
#                     promoter_labels.add(cleaned_label)
#                 elif feature.type == "rep_origin":
#                     identity = float(feature.qualifiers.get('identity', [0])[0])
#                     if identity >= 98:
#                         rep_origin_found = True
#                         rep_origin_identity = identity
#                         print(f"  Found origin of replication with identity {identity}%: {feature.qualifiers.get('label', ['Unknown'])[0]}")

#         print(f"CDS Labels: {cds_labels}")
#         print(f"Promoter Labels: {promoter_labels}")
#         print(f"Origin of Replication Found: {rep_origin_found} with Identity: {rep_origin_identity}%")

#     except Exception as e:
#         print(f"Error parsing {gbk_file}: {e}")
    
#     matching_labels = cds_labels.intersection(promoter_labels)
#     print(f"  Matching Labels: {matching_labels}")
    
#     return matching_labels, rep_origin_found, rep_origin_identity

# def process_files_in_directory(directory):
#     """Process all .gbk files in the given directory and find matching promoter-CDS names and check for origin of replication."""
#     summary = {}

#     for file_name in os.listdir(directory):
#         if file_name.endswith(".gbk"):
#             file_path = os.path.join(directory, file_name)
#             print(f"Processing file: {file_name}")
#             matching_labels, rep_origin_found, rep_origin_identity = analyze_gbk_file_for_matching_names_and_rep_origin(file_path)
#             if matching_labels or rep_origin_found:
#                 summary[file_name] = {
#                     'matching_labels': matching_labels,
#                     'rep_origin_found': rep_origin_found,
#                     'rep_origin_identity': rep_origin_identity
#                 }

#     return summary

# directory = "/home/stacys/src/masters_project/plasmids/5_generation_plannotate/annotations"
# summary = process_files_in_directory(directory)

# print("Plasmids with matching promoter and CDS names or origin of replication:")
# count = 0 
# for file_name, details in summary.items():
#     # if len(details['matching_labels']) > 0:
#     if details['rep_origin_found']:
#         print(f"{file_name}: Matching Labels={', '.join(details['matching_labels'])}, Origin of Replication Found={details['rep_origin_found']}, Identity={details['rep_origin_identity']}%")
#         count += 1

# print(count)
