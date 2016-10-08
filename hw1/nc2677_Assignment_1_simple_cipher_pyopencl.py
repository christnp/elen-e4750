"""
File: simple_cipher_pyopencl.py
Version: 1
Author: Nick Christma"
Description: This is a simple cipher written in PyOpenCL. This program accepts uppercase and lowercase inputs; 
			 however, it does not accept numberic valuse. The output will be in all uppercase."
Usage: Default input is 'Nick Christman.' Follow the script file name with a string of text to cipher any text."
"""
#---------- MAIN CODE ------------
# Import libraries
import platform # To get Python version
import string # ASCII characters
import time
import os # Used for integrating on Windows and Linux
import pyopencl as cl
import numpy as np
 
print "PyOpenCL Cipher v1.0"; 
print "Python ",platform.python_version();
 
#---------- PYTHON CODE ------------

# For right now, DO NOT have any spaces in the input 
#(use advanced_cipher_pyopencl.py to include spaces)
input_string = "nick";
str_dup_num = 1; # Easily adjust the length of the string by duplicating it this number of times
input_string = input_string*str_dup_num;

# Input string array
input_array = input_string.upper(); # Convert all characters to uppercase

# Cipher array
ascii_array = string.ascii_uppercase; # Store the ASCII characters for indexing

# Create/allocate the arrays to hold the input string array and the constant cipher array
input_array_np = np.array(input_array).astype(str);
ascii_array_np = np.array(ascii_array).astype(str);
output_array_np = np.empty_like(input_array_np);
#---------- END PYTHON ------------

#---------- PYOPENCL CODE ------------

# Define the kernel
kernel = """
__kernel void func(__global char* out, 
					__global char* in, 
					__constant char* ascii, 
					int in_size,
					int ac_size) 
{
	// PREPROCESSING
//#define DEBUG
	
	// START OF KERNEL
	
	// Get the global ID of the work item
    int inId = get_global_id(0);
	
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
		
		out[inId] = (in[inId] == ascii[j])?ascii[((elements)-j)]:out[inId];
		
	}
	
#ifdef DEBUG
	// Used for debugging kernel set-up
	if(inId < in_size){
		out[inId] = in[inId];
	}
#endif
	// END OF KERNEL
}
"""

# Select the desired OpenCL platform
NAME = 'NVIDIA CUDA';
platforms = cl.get_platforms();

devs = None;
platform_num = 0;
# Nice little routine to select and display the device that is being used
for platform in platforms:
	platform_num = platform_num + 1;	
	if platform.name == NAME:
		print "Using platform: ";
		print "     (",platform_num,") ", platform.name,"-",platform.version;
		devs = platform.get_devices();

if(devs == None):
	print "Platform ",NAME," was not found. Instead, using platform: ";
	print "     (",platform_num,") ", platform.name,"-",platform.version;
	devs = platform.get_devices();
	
# Set up a command queue:
ctx = cl.Context(devs)
queue = cl.CommandQueue(ctx)

# Create the memory buffers for the kernel; ensure proper memory flags (READ_ONLY, WRITE_ONLY, READ_WRITE)
mf = cl.mem_flags;
input_buf = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_array_np); #Input buffer
ascii_buf = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ascii_array_np); #ASCII buffer
outpt_buf = cl.Buffer(ctx,mf.WRITE_ONLY,output_array_np.nbytes);
#---------- END PYOPENCL ------------

print "\n--------------------";
print " INPUT STRING   ";
print "--------------------";
print ">> ",input_array_np;
print ">> ",input_array_np.nbytes,"bytes";

#---------- PROBLEMS ------------
#1. Input your name as an array of characters. Implement this simple cipher on Python (or C/C++).
print "\n\n ======================== PROBLEM 1 ========================\n";
pyth_output_array = [None]*(len(input_array)); # allocate memory for the output array
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

#2. Implement the same using: pyOpenCL/OpenCL
print "\n\n ======================== PROBLEM 2 ========================\n";
# Refer to the kernel and boilerplate OpenCL above.

# Launch the kernel. Since our kernel only accesses the global work item ID, 
# we simply set the local size to "None"
prg = cl.Program(ctx, kernel).build();
prg.func(queue, (input_array_np.nbytes,), None,
					outpt_buf, input_buf, ascii_buf,
					np.int32(input_array_np.nbytes), 
					np.int32(ascii_array_np.nbytes));
# Retrieve the results from the GPU using the output_array_np:
cl.enqueue_copy(queue, output_array_np, outpt_buf);
# Output the results
print "\n--------------------";
print " PYOPENCL OUTPUT   ";
print "--------------------";
print ">> ",output_array_np.astype(str);
print ">> ",output_array_np.nbytes,"bytes";
print "\n--------------------";					
					
#3. Repeat parts 1 & 2 with other string lengths (using a loop).
print "\n\n ======================== PROBLEM 3 ========================\n";
M = 5 # Number of overall iterations
N = 10 # Used for averaging
pyth_times_t = []
open_times_t = []
input_lapse = []
for i in range(M):
	# We have to change the input/output arrays and buffers each iteration
	input_string = "nick";
	input_string = input_string*((i+1)*2); #Double the total iteration

	# Input string array
	input_array = input_string.upper(); # Convert all characters to uppercase

	# Create/allocate the arrays to hold the input string array and the constant cipher array
	input_array_np = np.array(input_array).astype(str);
	output_array_np = np.empty_like(input_array_np);


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
	 
	open_times = []
	for j in range(N):
		start = time.time()
		#------ PYTHON ALGORITHM -----#    
		prg.func(queue, (input_array_np.nbytes,), None,
							outpt_buf, input_buf, ascii_buf,
							np.int32(input_array_np.nbytes),
							np.int32(ascii_array_np.nbytes));
		#------ END ALGORITHM -----#
		open_times.append(time.time()-start)
		
	pyth_times_t.append(np.average(pyth_times));
	open_times_t.append(np.average(open_times));
	
	print "\n >> ITERATION ",i+1,"<<";
	print "\n--------------------";
	print " INPUT STRING >> ",input_array_np,"(",input_array_np.nbytes,"bytes )";
	print "--------------------";
	
	print "\n--------------------";
	print " PYTHON OUTPUT   >> ",''.join(pyth_output_array),"(",len(pyth_output_array),"bytes )";	
	print " PYTHON TIME >> ",np.average(pyth_times)*1000,"ms @",M*N,"interations";
	print "--------------------";

	print "\n--------------------";
	print " PYOPENCL OUTPUT   >> ",output_array_np.astype(str),"(",output_array_np.nbytes,"bytes )";
	print " PYOPENCL TIME  >> ",np.average(open_times)*1000,"ms @",M*N,"interations"; 
	print "--------------------";

# Plot the result to a PNG using the matplotlib library
MAKE_PLOT = True
if MAKE_PLOT:
	m = np.arange(M);
	x = np.array(m).astype(int);
	y3_1 = [i * 1000 for i in pyth_times_t];
	y3_2 = [i * 1000 for i in open_times_t];
	
	import matplotlib as mpl
	mpl.use('agg')
	import matplotlib.pyplot as plt3

	plt3.gcf()	
	fig = plt3.figure()	
	cpu, = plt3.plot(x, y3_1,label="CPU")
	gpu, = plt3.plot(x, y3_2,label="GPU")
	plt3.legend(loc=2)
	plt3.xlabel('Number of Iterations')
	plt3.ylabel('Execution Time (ms)')  
	if(os.name == "nt"): # Windows FS
		filename = ''.join(np.array(["images\python_vs_opencl_Problem3_iter",M,".png"]));
	else: # All other FS (e.g., linux)
		filename = ''.join(np.array(["images/python_vs_opencl_Problem3_iter",M,".png"]));
	plt3.savefig(filename)
	print "\nSaved plot as \"",filename,"\"";
	plt3.close(fig)
	
#4. Determine at what number of letters does the GPU (OpenCl/CUDA) implementation time overcome 
#   the CPU-only (Python) implementation time.
print "\n\n ======================== PROBLEM 4 ========================\n";
M = 2 # Number of overall iterations; iterate twice, where the last one is to create a graphic
N = 10 # Used for averaging
pyth_times_tt = []
open_times_tt = []
num_letters_t = []
for iter in range(M):
	# Create empty arrays for storage	
	pyth_times_t = []
	open_times_t = []
	temp_array = []
	
	# Initialize variables
	string_length = 0;
	num_letters = 0;

	pyth_times = [0,0];
	open_times = [1,1];
	
	temp_array.append("a"); # Get the string started
	
	while np.average(pyth_times) < np.average(open_times)*(iter+1):
		# We have to change the input/output arrays and buffers each iteration			
		input_string = ''.join(temp_array)

		# Input string array
		input_array = input_string.upper(); # Convert all characters to uppercase

		# Create/allocate the arrays to hold the input string array and the constant cipher array
		input_array_np = np.array(input_array).astype(str);
		output_array_np = np.empty_like(input_array_np);
		

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
		 
		open_times = []
		for j in range(N):
			start = time.time()
			#------ PYTHON ALGORITHM -----#    
			prg.func(queue, (input_array_np.nbytes,), None,
								outpt_buf, input_buf, ascii_buf,
								np.int32(input_array_np.nbytes),
								np.int32(ascii_array_np.nbytes));
			#------ END ALGORITHM -----#
			open_times.append(time.time()-start)
			
		pyth_times_t.append(np.average(pyth_times));
		open_times_t.append(np.average(open_times));
				
		# Append the array by one character
		temp_array.append("x");
		num_letters+=1;
		# Print out for debugging
		#print num_letters," -- ",np.average(pyth_times)," -- ",np.average(open_times);
		
	pyth_times_tt.append(np.average(pyth_times_t));
	open_times_tt.append(np.average(open_times_t));
	num_letters_t.append(num_letters);

print "Python time at intersection \n >>",pyth_times_tt[0],"ms";
print "OpenCL time at intersection \n >>",open_times_tt[0],"ms";
print "Number of letters at intersection \n>>",num_letters_t[0];

# Plot the result to a PNG using the matplotlib library
MAKE_PLOT = True
if MAKE_PLOT:
	m = np.arange(1,(num_letters+1));
	x = np.array(m).astype(int);
	y4_1 = [i * 1000 for i in pyth_times_t];
	y4_2 = [i * 1000 for i in open_times_t];
	
	import matplotlib as mpl
	mpl.use('agg')
	import matplotlib.pyplot as plt4

	plt4.gcf()	
	fig = plt4.figure()
	plt4.plot(x, y4_1,label="CPU")
	plt4.plot(x, y4_2,label="GPU")
	plt4.legend(loc=2)
	plt4.xlabel('Number of Letters')
	plt4.ylabel('Execution Time (ms)')  
	if(os.name == "nt"): # Windows FS
		filename = ''.join(np.array(["images\python_vs_opencl_Problem4_",num_letters,"_iter",M,".png"]));
	else: # All other FS (e.g., linux)
		filename = ''.join(np.array(["images/python_vs_opencl_Problem4_",num_letters,"_iter",M,".png"]));
	plt4.savefig(filename)
	print "\nSaved plot as \"",filename,"\"";
	plt4.close(fig)
#---------- END PROBLEMS ------------

#---------- END MAIN ------------	