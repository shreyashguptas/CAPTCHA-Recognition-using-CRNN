import os

# Get the current directory
current_directory = os.getcwd()

# File extensions to search for
extensions = ['.html', '.css', '.js', '.py']

# Output file name
output_file = 'combined_code.txt'

# Open the output file in write mode
with open(output_file, 'w') as outfile:
    # Iterate over each file in the current directory
    for filename in os.listdir(current_directory):
        # Check if the file has one of the desired extensions
        if any(filename.endswith(ext) for ext in extensions):
            # Open the file and read its contents
            with open(filename, 'r') as infile:
                content = infile.read()
                
                # Write the file type and content to the output file
                outfile.write(f"--- {filename} ---\n")
                outfile.write(content)
                outfile.write("\n\n")

print(f"Combined code saved to {output_file}")