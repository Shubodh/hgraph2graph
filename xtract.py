def extract_first_n_lines(input_file, output_file, n):
    with open(input_file, 'r') as infile:
        lines = [next(infile) for _ in range(n)]  # Read the first n lines

    with open(output_file, 'w') as outfile:
        outfile.writelines(lines)  # Write those lines to the output file

# Usage
input_file = 'data/chembl/all.txt'  # Replace with your input file name
output_file = 'data/chembl/all_small.txt'  # Replace with your desired output file name
n = 150000  # Change this to the number of lines you want to extract

extract_first_n_lines(input_file, output_file, n)
