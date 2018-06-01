A note on the files.
	in_cou.txt contains new line separated input file locations (absolute or relative to local)
	ou_cou.txt contains new line separated output file locations (absolute or relative to local)
		These two files are must be registered.
			the image file referenced by line 1 of in_cou.txt is decoupled and OVERWRITES the file name referenced by line 1 of ou_cou.txt
	debug.txt countains run time info for the most recent run
	mem.txt contains the side size of array to be cut (currently 256 was 512 but change to double precision cut it) must be 2**n.  Larger runs faster. 256 is limit on current system.
	size.txt contains the side size of the full image.  Must be multiple of 512.
	look_up_true_large.txt is the lookup table referenced by the decoupling technique.
		it is formatted in the following way
	1: 	\t	S0\t	S1\t	S2\t	...	Sk\n
	2:	N0\t	A00\t	A01\t	A02\t	...	A0k\n
	3:	N1\t	A10\t	A11\t	A12\t	...	A1k\n
	.	.	.	.	.	...	.
	.	.	.	.	.	...	.
	.	.	.	.	.	...	.
	j+1:	Nj\t	Aj0\t	Aj1\t	.	...	Ajk\n

	S is the level of the central pixel
	N is the level of the adjacent pixel
	A is the coupling coefficient (IN PERCENT)
	look at ../Support_Functions/build_table.py to see an example of how this file is generated using a two dimensional function.
	right now in_frame.csv is an input image.  running For_Git.exe will generate [or overwrite] out_frame.csv 
	For_Git.exe runs on a GTX960M system with CUDA installed.
		*.dll, *.ilk, *.pdb let the *.exe run
	

