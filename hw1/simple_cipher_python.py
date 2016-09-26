"""
1. (10 pts) Input your name as an array of characters. Implement this simple cipher on Python (or C/C++).

File: simple_cipher_python.py"
Version: 1"
Author: Nick Christman"
Description: This is a simple cipher written in Python. This program accepts uppercase and lowercase inputs; however, it does not accept numberic valuse. The output will be in all uppercase."
Usage: Default input is 'Nick Christman.' Follow the script file name with a string of text to cipher any text."
"""

#---------- MAIN CODE ------------
# Import libraries
import string; # ASCII characters
import sys;	# Input arguments

# Input string
input_arg_len = len(sys.argv);
if input_arg_len > 1: # first argument is the python script
	input = [None]*(input_arg_len-1); # allocate memory for the array
	for i in range(input_arg_len-1):		
		input[i] = sys.argv[i+1]; # We are only interested in the inputs after the file name
	input = ' '.join(input);
else:
	input = "Nick Christman";
	

print "Original input string: ", input;

# Input array
input = input.upper(); # Convert all characters to uppercase
input_len = len(input); # This includes spaces
in_array = input.split(); # We need the input to be an array so we can step through each element
in_array_len = len(in_array); # Number of groups of characters


# Cipher array
ascii_array = list(string.ascii_uppercase);
ascii_len = len(ascii_array);
cipher = ascii_array[::-1]; #reverse the ascii code list to match our desired cipher

# Encrypt the input
output_array = [None]*in_array_len; # allocate memory for the output array

for i in range(in_array_len): #Loop through the number of space seperated strings
	# Create some temporary variables
	temp_list = list(in_array[i]); # turn the iTH input string into a list/array
	temp_list_len = len(temp_list);
	temp_output = [None]*temp_list_len;
	
	for j in range(temp_list_len): # Loop through each character of the iTH string
		temp_index = ascii_array.index(temp_list[j]); # Index of the temp_list character within the ascii array		
		temp_output[j] = cipher[temp_index]; # Use that index to determine the new cipher character
		#--- end for j
		
	# iTH output string is encrypted	
	output_array[i] = ''.join(temp_output); # Join the list into a single string
#--- end for i

output = ' '.join(output_array);

print "\n******************** MAIN OUTPUT ********************"
print "Ciphered input string: ", output;
		