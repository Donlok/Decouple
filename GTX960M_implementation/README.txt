A note on the files.
	in_cou.txt contains new line separated input file locations (absolute or relative to local) 
	ou_cou.txt contains new line separated output file locations (absolute or relative to local)
		The file names are *.fits
		These two files are must be registered.
			the image file contained in line i of in_cou.txt is decoupled and saved to the file name in line i of ou_cou.txt. FAILS IF OUTPUT FILE ALREADY EXISTS 
	debug.txt countains run time info for the most recent run
	mem.txt contains the side size of array to be cut must be 2**n.  Larger runs faster. 512 is limit on current system.  Limit depends on size of look_up.txt as well.
	size.txt contains the side size of the full image.  Must be multiple of 512.
	i_count.txt contains the number of iterations to be used.  Recommend 5
	msca.txt contains a 1 or 0.
		1 indicates that the coupling is not identical in all four directions (up down left and right).
			this requires the presence of look_up_1.txt(up), look_up_2.txt(down), look_up_3.txt(left), look_up_4.txt(right), each a lookup table of properly formatted SNA data
		0 indicates that the coupling is identical in all directions and only a single table will be used
			this requires the presence of look_up.txt
	look_up.txt and/or look_up_n.txt is the lookup table referenced by the decoupling technique.
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

	For_Git.exe runs on a GTX960M system with CUDA installed.
		*.dll, *.ilk, *.pdb let the *.exe run
	

