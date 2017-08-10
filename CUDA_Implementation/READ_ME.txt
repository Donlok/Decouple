for_git.cpp is the source code file
in_cou.txt is a text file containing the location and filenames of any input files
	at the moment it links to ./random_0RMS.csv
look_up_tru_large.txt is a text file containing the look up table of alpha values to be referenced
ou_cou.txt is a text file containing the location and filenames of any output files
	at the moment it links to ./decoupled_random_0RMS.csv
	it must contain the same number of links as in_cou.txt
random_0RMS.csv is a sample frame
size.txt is a text file containing a single integer that is the side length of the square input frame
	currently 128

For_Git.exe, cudart32_75.dll, and cudart64_75.dll are the linked files from a compiled version on a particular system
running For_Git.exe (or a locally compiled executable with access to apropriate *.dlls)  results in opening of a dialogue box which gives timing information as the deconvolution runs.
Afer the run completes, decoupled_random_0RMS.csv will be created containing the decoupled image file and debug.txt will be created containing information on the number of files and total run time of the program.