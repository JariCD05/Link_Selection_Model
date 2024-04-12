

# Assuming performance_parameters_over_time is the dictionary you obtained from the modified function
def export_performance_parameters_to_scsv(performance_parameters_over_time, filename="performance_parameters_over_time.csv"):
    with open(filename, 'w') as file:
        # Write the header
        file.write("Parameter Index;Time Step;Satellite Values\n")
        
        # Iterate through each performance parameter
        for param_index, time_series in performance_parameters_over_time.items():
            for time_step, values in enumerate(time_series):
                # Convert numpy array to a semicolon-separated string
                values_str = ";".join(map(str, values))
                # Write the parameter index, time step, and satellite values to the file
                file.write(f"{param_index};{time_step};{values_str}\n")

# Example call to the export function
#export_performance_parameters_to_scsv(performance_parameters_over_time)

#print(f"Exported performance parameters to CSV file.")


import pandas as pd

# Assuming performance_parameters_over_time is the dictionary with your data
def export_performance_parameters_to_csv(performance_parameters_over_time, filename="performance_parameters_over_time_adjusted.csv"):
    # Define the weights
    weights = {
        'Availability': 0.2,
        'BER': 0,
        'Cost': 0.2,
        'Latency': 0.2,
        'Throughput': 0.4
    }
    
    # Prepare the CSV content
    headers = ['Timestamp', 'Satellite', 'Availability', 'BER', 'Cost', 'Latency', 'Throughput', 'Weighted Availability', 'Weighted BER', 'Weighted Cost', 'Weighted Latency', 'Weighted Throughput', 'Total Weighted Sum']
    rows = []
    
    # Iterate over each time step
    for time_step in range(len(performance_parameters_over_time[0])):
        for sat_index in range(len(performance_parameters_over_time[0][time_step])):
            row = [time_step, f"Sat {sat_index+1}"]
            total_weighted_sum = 0
            for param_index, param in enumerate(['Availability', 'BER', 'Cost', 'Latency', 'Throughput']):
                value = performance_parameters_over_time[param_index][time_step][sat_index]
                if np.isnan(value):
                    row.append('NA')
                    row.append('NA')  # For weighted value as well
                else:
                    row.append(value)
                    weighted_value = value * weights[param]
                    row.append(weighted_value)
                    total_weighted_sum += weighted_value
            row.append(total_weighted_sum)
            rows.append(row)
        
        # Add a row for weights
        rows.append(['Weights', ''] + list(weights.values()) + [''] * 6)
    
    # Convert to DataFrame for easier CSV export
    df = pd.DataFrame(rows, columns=headers)
    # Export to CSV
    df.to_csv(filename, index=False, sep=';')

    return filename

# Example function call (assuming you have the data in the appropriate structure)
filename = export_performance_parameters_to_csv(performance_parameters_over_time)

filename

import pandas as pd

# Load the CSV data
df = pd.read_csv('performance_parameters_over_time_adjusted.csv', sep=';')

# Initialize the LaTeX document
latex_document = """
\\documentclass{article}
\\usepackage{booktabs}
\\begin{document}
\\title{Satellite Performance Parameters}
\\author{Author Name}
\\date{\\today}
\\maketitle
"""



# Iterate through each timestamp
for timestamp in sorted(df['Timestamp'].dropna().unique()):
    # Filter the DataFrame for the current timestamp
    df_timestamp = df[df['Timestamp'] == timestamp]
    
    # Start the table for this timestamp
    latex_table = f"""
\\section*{{Performance Parameters at Timestamp {int(timestamp)}}}
\\begin{"table"}[ht]
\\centering
\\caption{{Performance metrics and weighted sums for each satellite at Timestamp {int(timestamp)}.}}
\\begin{"tabular"}{{@{{}}lccccccccccc@{{}}}}
\\toprule
Satellite & Availability & BER & Cost & Latency & Throughput & W. Availability & W. BER & W. Cost & W. Latency & W. Throughput & Total W. Sum \\\\ \\midrule
"""
    # Add rows for each satellite
    for _, row in df_timestamp.iterrows():
        latex_table += f"{row['Satellite']} & {row['Availability']} & {row['BER']} & {row['Cost']} & {row['Latency']} & {row['Throughput']} & {row['Weighted Availability']} & {row['Weighted BER']} & {row['Weighted Cost']} & {row['Weighted Latency']} & {row['Weighted Throughput']} & {row['Total Weighted Sum']} \\\\\n"
    
    # Close the table
    latex_table += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    
    # Add the table for this timestamp to the document
    latex_document += latex_table

# Close the LaTeX document
latex_document += "\\end{document}\n"

# For demonstration here, we're showing a part of the LaTeX document. 
# You should write this to a file as follows:
# with open('satellite_performance_tables.tex', 'w') as f:
#     f.write(latex_document)
print(latex_document[:1000])  # Printing only the first 1000 characters for demonstration


