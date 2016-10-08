
"""
File: simple_cipher_PYCUDA.py
Version: 1
Author: Nick Christman
Description: This is a simple cipher written in PYCUDA. This program accepts uppercase and lowercase inputs; 
			 however, it does not accept numberic valuse. The output will be in all uppercase.
Usage: Default input is 'Nick Christman.' Follow the script file name with a string of text to cipher any text.
"""
#---------- MAIN CODE ------------
# Import libraries
import platform # To get Python version
import string # ASCII characters
import math
import time
import os # Used for integrating on Windows and Linux
import numpy as np
from pycuda import driver, compiler, gpuarray, tools

# Initialize the device
import pycuda.autoinit

print "PYCUDA Cipher v1.0";
print "Python ",platform.python_version();

#---------- PYTHON CODE ------------

# For right now, DO NOT have any spaces in the input 
#(use advanced_cipher_PYCUDA.py to include spaces)
input_string = "nick";
str_dup_num = 1; # Easily adjust the length of the string by duplicating it this number of times
input_string = input_string*str_dup_num;

# Input string array
input_array = input_string.upper(); # Convert all characters to uppercase

# Cipher array
ascii_array = string.ascii_uppercase; # Store the ASCII characters for indexing

# Create/allocate the arrays to hold the input string array and the constant cipher array
input_array_np = np.array(input_array,dtype=str);
ascii_array_np = np.array(ascii_array,dtype=str);
output_array_np = np.zeros_like(input_array_np,dtype=np.long);

# Define the input sizes
input_size = input_array_np.nbytes;
ascii_size = ascii_array_np.nbytes;
#---------- END PYTHON ------------

#---------- PYCUDA CODE ------------
dev = pycuda.autoinit.device;
print dev.name(),"with compute capability of",dev.compute_capability(),"\n";

# Define the kernel
cipher_kernel  = """
__global__ void cipherFunc(char* out, 
					 char* in, 
					 char* ascii, 
					 int in_size,
					 int ac_size) 
{
	// PREPROCESSING
//#define DEBUG
	
	// START OF KERNEL
	
	// 1D thread index
	int inIdx = blockDim.x * blockIdx.x + threadIdx.x;
	
	
	// Declare some local variables.
	int j;
	int elements = ac_size-1;	// The number of bytes in the ASCII array. We 
								// subtract "1" to acount for the fact we 
								// start counting at "0"
	
	// On each character of the input char, loop through the cipher
	// char array and store that cipher character in the output char
	// if there is a match. If there is not a match, then do not change
	// the output char.
	for(j=0;j<ac_size;j++){	
	
		// Compare the input character to each character of the ASCII array. 
		// When we have a match, we simply take the inverted index of the 
		// current position in the ASCII array (ascii[((elements)-j)]). This 
		// gives us the correct cipher element for the input text.
		
		out[inIdx] = (in[inIdx] == ascii[j])?ascii[((elements)-j)]:out[inIdx];
		
	}
	
#ifdef DEBUG
	// Used for debugging kernel set-up
	if(inIdx < in_size){
		out[inIdx] = in[inIdx];
	}
#endif
	// END OF KERNEL
}
"""

# Configure device dimensions
block_size = input_size;
block = (block_size, 1, 1); # For a string of characters, we are assuming a 1D Block and 1D Grid
grid_size = 32;
grid = grid_size;


# Transfer host memory to device memory
input_gpu = gpuarray.to_gpu(input_array_np);
ascii_gpu = gpuarray.to_gpu(ascii_array_np);
output_gpu = gpuarray.empty((block_size,1),dtype=np.uint8); #STR in python is a long (32-bit)

# Get the kernel code from the template by specifying the constant size
kernel_code = cipher_kernel % {
    'block': block
	};

# Compile the kernel code
mod = compiler.SourceModule(kernel_code);
 
# Get the kernel function from the compiled module
cipher = mod.get_function("cipherFunc");
 
#---------- END PYCUDA ------------

print "\n--------------------";
print " INPUT STRING   ";
print "--------------------";
print ">> ",input_array_np;
print ">> ",input_array_np.nbytes,"bytes";

#---------- PROBLEMS ------------
#1. Input your name as an array of characters. Implement this simple cipher on Python (or C/C++).
print "\n\n ======================== PROBLEM 1 ========================\n";
pyth_output_array = [None]*(len(input_array)); # Allocate memory for the output array
for i in range(len(input_array)): #Loop through the number of space seperated strings
	
	for j in range(len(ascii_array)): # Loop through each character of the iTH string
		temp_index = ascii_array.index(input_array[i]); # Index of the temp_list character within the ascii array		
			
		pyth_output_array[i] = ascii_array[(len(ascii_array)-temp_index)-1]; # Use that index to determine the new cipher character		
		#--- end for j	
		
#--- end for i
	
# Output the results
print "\n--------------------";
print " PYTHON OUTPUT   ";
print "--------------------";
print ">> ",''.join(pyth_output_array);
print ">> ",len(pyth_output_array),"bytes";
print "\n--------------------";

#2. Implement the same using: pyCuda/Cuda
print "\n\n ======================== PROBLEM 2 ========================\n";
# Refer to the kernel and boilerplate Cuda above.

# Call the kernel on the card
cipher(output_gpu, input_gpu, ascii_gpu,
		np.int32(input_size), np.int32(ascii_size), block=block);

# Get the result data from the device and convert to ASCII
output_array_np = [chr(item) for item in output_gpu.get()];

# Conver the array of characters to a string
output_array = np.str("".join(output_array_np));

# Output the results
print "\n--------------------";
print " PYCUDA OUTPUT   ";
print "--------------------";
print ">> ",output_array;
print ">> ",len(output_array),"bytes";
print "\n--------------------";					
					
#3. Repeat parts 1 & 2 with other string lengths (using a loop).
print "\n\n ======================== PROBLEM 3 ========================\n";
M = 5 # Number of overall iterations
N = 10 # Used for averaging
pyth_times_t = []
cuda_times_t = []
input_lapse = []
for i in range(M):
	# We have to change the input/output arrays and buffers each iteration
	input_string = "nick";
	input_string = input_string*((i+1)*2); #Double the total iteration

	# Input string array
	input_array = input_string.upper(); # Convert all characters to uppercase

	# Create/allocate the arrays to hold the input string array and the constant cipher array
	input_array_np = np.array(input_array).astype(str);
	output_array_np = np.zeros_like(input_array_np,dtype=np.long);
	input_size = input_array_np.nbytes;
	
	# Re-configure GPU based on input size
	block_size = input_size;
	block = (block_size, 1, 1); # For a string of characters, we are assuming a 1D Block and 1D Grid
	input_gpu = gpuarray.to_gpu(input_array_np);
	output_gpu = gpuarray.empty((block_size,1),dtype=np.uint8); #STR in python is a long (32-bit)

	pyth_times = []
	for j in range(N):
		start = time.time();
		#------ PYTHON ALGORITHM -----#
		pyth_output_array = [None]*(len(input_array)); # allocate memory for the output array
		for m in range(len(input_array)): #Loop through the number of space seperated strings
			
			for n in range(len(ascii_array)): # Loop through each character of the iTH string
				temp_index = ascii_array.index(input_array[m]); # Index of the temp_list character within the ascii array		
				
				
				pyth_output_array[m] = ascii_array[(len(ascii_array)-temp_index)-1]; # Use that index to determine the new cipher character		
				#--- end for j	
				
		#--- end for i

		#------ END ALGORITHM -----#
		pyth_times.append(time.time()-start);
	 
	cuda_times = []
	for j in range(N):
		start = time.time()
		#------ PYCUDA ALGORITHM -----#    
		# Call the kernel on the card
		cipher(output_gpu, input_gpu, ascii_gpu,
				np.int32(input_size), np.int32(ascii_size), block=block);
		# #------ END ALGORITHM -----#
		cuda_times.append(time.time()-start)
	
	pyth_times_t.append(np.average(pyth_times));
	cuda_times_t.append(np.average(cuda_times));
	
	
	# Get the result data from the device and convert to ASCII
	output_array_np = [chr(item) for item in output_gpu.get()];

	# Conver the array of characters to a string
	output_array = np.str("".join(output_array_np));
	
	print "\n >> ITERATION ",i+1,"<<";
	print "\n--------------------";
	print " INPUT STRING >> ",input_array_np,"(",input_array_np.nbytes,"bytes )";
	print "--------------------";
	
	print "\n--------------------";
	print " PYTHON OUTPUT   >> ",''.join(pyth_output_array),"(",len(pyth_output_array),"bytes )";	
	print " PYTHON TIME >> ",np.average(pyth_times)*1000,"ms @",M*N,"interations";
	print "--------------------";

	print "\n--------------------";
	print " PYCUDA OUTPUT   >> ",output_array,"(",len(output_array),"bytes )";
	print " PYCUDA TIME  >> ",np.average(cuda_times)*1000,"ms @",M*N,"interations"; 
	print "--------------------";

# Plot the result to a PNG using the matplotlib library
MAKE_PLOT = True
if MAKE_PLOT:
	m = np.arange(M);
	x = np.array(m).astype(int);
	y3_1 = [i * 1000 for i in pyth_times_t];
	y3_2 = [i * 1000 for i in cuda_times_t];
	
	import matplotlib as mpl
	mpl.use('agg')
	import matplotlib.pyplot as plt3

	plt3.gcf()	
	fig = plt3.figure()	
	cpu, = plt3.plot(x, y3_1,label="CPU")
	gpu, = plt3.plot(x, y3_2,label="GPU")
	plt3.legend(loc=2)
	#plt.xticks(m)
	plt3.xlabel('Number of Iterations')
	plt3.ylabel('Execution Time (ms)')  
	if(os.name == "nt"): # Windows FS
		filename = ''.join(np.array(["images\python_vs_cuda_Problem3_iter",M,".png"]));
	else: # All other FS (e.g., linux)
		filename = ''.join(np.array(["images/python_vs_cuda_Problem3_iter",M,".png"]));
	plt3.savefig(filename)
	print "\nSaved plot as \"",filename,"\"";
	plt3.close(fig)
	
#4. Determine at what number of letters does the GPU (cuda/CUDA) implementation time overcome 
#   the CPU-only (Python) implementation time.
print "\n\n ======================== PROBLEM 4 ========================\n";
M = 3 # Number of overall iterations; iterate twice, where the last one is to create a graphic
N = 100 # Used for averaging
pyth_times_tt = []
cuda_times_tt = []
num_letters_t = []
for iter in range(M):
	# Create empty arrays for storage	
	pyth_times_t = []
	cuda_times_t = []
	temp_array = []
	
	# Initialize variables
	string_length = 0;
	num_letters = 0;

	pyth_times = [0,0];
	cuda_times = [1,1];
	
	temp_array.append("a"); # Get the string started
	
	while np.average(pyth_times) < np.average(cuda_times)*(iter+1):
		# We have to change the input/output arrays and buffers each iteration		
		
		input_string = ''.join(temp_array)
		
		#input_string = input_string*(string_length++); #Double the total iteration

		# Input string array
		input_array = input_string.upper(); # Convert all characters to uppercase

		# Create/allocate the arrays to hold the input string array and the constant cipher array
		input_array_np = np.array(input_array).astype(str);
		output_array_np = np.zeros_like(input_array_np,dtype=np.long);
		input_size = input_array_np.nbytes;
		
		# Re-configure GPU based on input size
		block_size = input_size;
		block = (block_size, 1, 1); # For a string of characters, we are assuming a 1D Block and 1D Grid
		input_gpu = gpuarray.to_gpu(input_array_np);
		output_gpu = gpuarray.empty((block_size,1),dtype=np.uint8); #STR in python is a long (32-bit)

		pyth_times = []
		for j in range(N):
			start = time.time();
			#------ PYTHON ALGORITHM -----#
			pyth_output_array = [None]*(len(input_array)); # allocate memory for the output array
			for m in range(len(input_array)): #Loop through the number of space seperated strings
				
				for n in range(len(ascii_array)): # Loop through each character of the iTH string
					temp_index = ascii_array.index(input_array[m]); # Index of the temp_list character within the ascii array		
					
					
					pyth_output_array[m] = ascii_array[(len(ascii_array)-temp_index)-1]; # Use that index to determine the new cipher character		
					#--- end for j	
					
			#--- end for i

			#------ END ALGORITHM -----#
			pyth_times.append(time.time()-start);
		 
		cuda_times = []
		for j in range(N):
			start = time.time()
			#------ PYCUDA ALGORITHM -----#    
			# Call the kernel on the card
			cipher(output_gpu, input_gpu, ascii_gpu,
					np.int32(input_size), np.int32(ascii_size), block=block);
			# #------ END ALGORITHM -----#
			cuda_times.append(time.time()-start)
			
		pyth_times_t.append(np.average(pyth_times));
		cuda_times_t.append(np.average(cuda_times));
				
		# Append the array by one character
		temp_array.append("x");
		num_letters+=1;
		# Print out for debugging
		#print num_letters," -- ",np.average(pyth_times)," -- ",np.average(cuda_times);
		
	pyth_times_tt.append(np.average(pyth_times_t));
	cuda_times_tt.append(np.average(cuda_times_t));
	num_letters_t.append(num_letters);

print "Python time at intersection \n >>",pyth_times_tt[0],"ms";
print "cuda time at intersection \n >>",cuda_times_tt[0],"ms";
print "Number of letters at intersection \n>>",num_letters_t[0];

# Plot the result to a PNG using the matplotlib library
MAKE_PLOT = True
if MAKE_PLOT:
	m = np.arange(1,(num_letters+1));
	x = np.array(m).astype(int);
	y4_1 = [i * 1000 for i in pyth_times_t];
	y4_2 = [i * 1000 for i in cuda_times_t];
	
	import matplotlib as mpl
	mpl.use('agg')
	import matplotlib.pyplot as plt4

	plt4.gcf()	
	fig = plt4.figure()
	plt4.plot(x, y4_1,label="CPU")
	plt4.plot(x, y4_2,label="GPU")
	plt4.legend(loc=2)
	plt4.xticks(m)
	plt4.xlabel('Number of Letters')
	plt4.ylabel('Execution Time (ms)')  
	if(os.name == "nt"): # Windows FS
		filename = ''.join(np.array(["images\python_vs_cuda_Problem4_",num_letters,"_iter",M,".png"]));
	else: # All other FS (e.g., linux)
		filename = ''.join(np.array(["images/python_vs_cuda_Problem4_",num_letters,"_iter",M,".png"]));
	plt4.savefig(filename)
	print "\nSaved plot as \"",filename,"\"";
	plt4.close(fig)
#---------- END PROBLEMS ------------

#---------- END MAIN ------------