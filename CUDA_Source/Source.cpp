#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <chrono>			//current time
#include <ctime>			//timing
#include <iostream>			//io management
#include <iomanip>			//precision management for io.  Control for making of csvs.
#include <stdio.h>			//printf
#include <cmath>			//some math
#include <limits>			//float rounding
#include <fstream>			//file manipulation
#include <string>			//strings
#include <tuple>			//tuples
#include <sstream>			//string streams
#include <vector>			//vectors

//global variables
int*max_size;
bool have_we_allocated = false;
bool memory_check = false;
bool v = true;
int*GPUmemoryconstraint;	//Keep it a power of two.  This allows for adaptation to GPUs of different sizes

//for recombination of processes
//contains data from a start point of a given size
struct SubArray{
	double** values = NULL;
	int startrow = NULL;
	int startcol = NULL;
	int rows = NULL;
	int cols = NULL;
};

//Global function declarations
void print_time(std::clock_t t2, std::clock_t t1);
void print_line(double*A, int size);
void print_array(double**A, int col, int row);
void deconstruct_subarray(SubArray& B);
void array_to_subarray(double** A, SubArray& B, int startcol, int startrow, int cols, int rows);
void check(int a);
void init_array(double**& A, int col, int row);

//This removes the contents from subarray.values and allows for reutilization of the same subarray again.
void deconstruct_subarray(SubArray& B){
	for (int i = 0; i < B.rows; i++){
		delete[] B.values[i];
	}
	delete B.values;
	B.values = NULL;
	B.startrow = NULL;
	B.startcol = NULL;
	B.rows = NULL;
	B.cols = NULL;
}

//this deconstructs a line array and reinitialize the first vector to NULL
void deconstruct_line(double*& A){
	delete A;
	A = NULL;
}

// this deconstructs a float line and reinitialize the first vector to NULL

void deconstruct_float_line(float*& A){
	delete A;
	A = NULL;
}

void deconstruct_array(double **& A, int rows){
	int i = 0;
	while (i < rows){
		delete[] A[i];
		i++;
	}
	delete A;
	A = NULL;
}

void deconstruct_tuple(std::tuple<double**, int, int>& A){
	int i = 0;
	int rows = std::get<2>(A);
	while (i < rows){
		delete[] std::get<0>(A)[i];
		i++;
	}
	delete std::get<0>(A);
	std::get<0>(A) = NULL;
}

//places data from an array into a subarray starting at point startcol,startrow of size row, col
//Subarray must be initialized but empty.
void array_to_subarray(double** A, SubArray& B, int startcol, int startrow, int cols, int rows){
	B.values = new double*[rows];
	int m = 0;
	for (int i = startrow; i < (startrow + rows); i++){
		int n = 0;
		B.values[m] = new double[cols];
		for (int j = startcol; j < (startcol + cols); j++){
			B.values[m][n] = A[i][j];
			n++;
		}
		m++;
	}
	B.cols = cols;
	B.rows = rows;
	B.startcol = startcol;
	B.startrow = startrow;
}

// Places values from a subarray into a pre-existing double**
// double** must have sufficient size.  the function will never throw any errors but will not include data outside its initialized dimensions
void subarray_into_array(double**& A, SubArray B, int rows, int cols){
	for (int i = B.startrow; (i < B.startrow + B.rows) && (i < rows); i++){
		for (int j = B.startcol; (j < B.startcol + B.cols) && (j < cols); j++){
			if ((B.values[i - B.startrow][j - B.startcol] == B.values[i - B.startrow][j - B.startcol]) && !(abs(B.values[i - B.startrow][j - B.startcol])>100000.0)){
				A[i][j] = B.values[i - B.startrow][j - B.startcol];
			}
		}
	}
}

//function for debugging, prints specified integer.
void check(int i){
	std::cout << "Checkpoint " << i << " has been reached." << std::endl;
}

// splits a string using vectors.  This is slow but is general for sizes.
// This is currently used in building the array for the look-up table
void split_string(std::string line, char delim, std::vector<std::string>& broken){
	std::stringstream ss(line);
	std::string item;
	while (getline(ss, item, delim)){
		broken.push_back(item);
	}
}

// splits string using arrays.  This is about a factor of two faster but requires arrays be initialized of known size.
// This is currently used in building the array from the .csv file.
int split_string_faster(std::string line, char delim, std::string*& broken){
	std::stringstream ss(line);
	std::string item;
	int i = 0;
	while (getline(ss, item, delim) && (i<*max_size)){
		broken[i] = item;
		i++;
	}
	return i;
}

// saves data from an array into a *.csv file
// *.csv is ready for conversion to python numpy array using numpy.read()
void array_to_csv(double** A, int col, int row, std::string fname){
	std::ofstream ofile;
	std::stringstream ss;
	std::string str;
	ofile.open(fname);
	for (int i = 0; i < row; i++){
		for (int j = 0; j < col; j++){
			ss << std::fixed << /* std::setfill('0') << std::setw(9) << */ std::setprecision(4) << A[i][j];
			str = ss.str();
			ss.str(std::string());
			ofile << str;
			if (j != col - 1){ ofile << ","; }
		}
		if (i != row - 1){ ofile << std::endl; }
	}
	ofile.close();
}

// replaces the contents of array A with the difference between array A and B
// requires array A and array B both contain data from [0,row)[0,col)
void array_diff_mag(double**&A, double**B, int col, int row){
	for (int i = 0; i < row; i++){
		for (int j = 0; j < col; j++){
			A[i][j] = abs(A[i][j] - B[i][j]);
		}
	}
}


// converts a .csv file into a usable datatype
// returns a tuple which contains the array, the row size, and the col size.
// no longer utilizes a temp array
// This function is the bottleneck of this program at the moment
std::tuple<double**, int, int> csv_to_array(std::string fname, int imin, int imax, int jmin, int jmax){
	std::clock_t preconver = std::clock();
	double**A;
	std::ifstream infile(fname);
	if (!infile.is_open()) {
		std::cout << " Failed to open" << std::endl;
		double** Q = NULL;
		return std::make_tuple(Q, 0, 0);
	}
	std::string line;
	std::string** temp = new std::string*[*max_size];
	int num_cols = jmax - jmin;
	int num_rows = imax - imin;
	int i = 0;
	int templen = 0;
	while (std::getline(infile, line) && i<imax){
		temp[i] = new std::string[*max_size];
		templen++;
		int row_size = split_string_faster(line, ',', temp[i]);
		if (i == 0 && num_cols>row_size){ num_cols = row_size; }
		i++;
	}
	i = 0;
	A = new double*[imax - imin];
	while ((i < *max_size) && (i<(imax - imin))){
		int j = 0;
		A[i] = new double[jmax - jmin];
		while (j < num_rows && j<(jmax - jmin)){
			A[i][j] = std::stod(temp[i + imin][j + jmin]);
			j++;
		}
		i++;
	}
	i = 0;
	while (i < templen){
		delete[] temp[i];
		i++;
	}
	delete temp;
	temp = NULL;

	return std::make_tuple(A, num_cols, num_rows);
}

// converts from a textfile into a tuple.  The tuple contains the old SCA data and the size of A in cols and rows
// double** A contains alpha data
// C contains 'contrast' data, just one look up dimension for alpha.  This naming convention is a relic.  'C' data now contains 'N' values (from contrast to neighbor)
// S contains 'signal' data, anohter lookup dimension for alpha
// This could be the bottleneck for very large text files.
std::tuple<float**, float*, float*, int, int> build_table(std::string fname){
	std::ifstream infile(fname);
	std::string line;
	std::vector<std::string> col;
	std::vector<std::string> temp_C;
	std::vector<std::vector<std::string>> temp_A;
	std::vector<std::string> temp_A_line;
	float** A;
	float* S;
	float* C;
	int Ssize;
	int Csize;
	int i = 0;
	while (std::getline(infile, line)){
		split_string(line, '\t', col);
		if (i == 0){
			col.erase(col.begin());
			S = new float[col.size()];
			Ssize = col.size();
			int j = 0;
			while (j < Ssize){
				S[j] = std::stof(col[j]);
				j++;
			}
		}
		else{
			temp_C.push_back(col[0]);
			col.erase(col.begin());
			int j = 0;
			while (j < col.size()){
				temp_A_line.push_back(col[j]);
				j++;
			}
			temp_A.push_back(temp_A_line);
			temp_A_line.clear();
		}
		col.clear();
		i++;
	}
	Csize = temp_C.size();
	C = new float[temp_C.size()];
	int j = 0;
	while (j < temp_C.size()){
		C[j] = std::stof(temp_C[j]);
		j++;
	}
	temp_C.clear();
	i = 0;
	A = new float*[temp_A.size()];
	while (i < temp_A.size()){
		j = 0;
		A[i] = new float[temp_A[i].size()];
		while (j < temp_A[i].size()){
			A[i][j] = std::stof(temp_A[i][j]);
			j++;
		}
		i++;
	}
	return std::make_tuple(A, S, C, Ssize, Csize);
}

// initializes a line array of floats
void init_float_line(float*&A, int size){
	A = new float[size];
}

// initializes a line array of doubles
void init_line(double*&A, int size){
	A = new double[size];
}

//initializes a two dimensional array of doubles
void init_array(double**&A, int col, int row){
	A = new double*[row];
	for (int i = 0; i < row; i++){
		A[i] = new double[col];
	}
}

//fills a line with values from offset counting upward
void fill_line(double*&A, int size, int offset){
	for (int i = 0; i < size; i++){
		A[i] = i + offset;
	}
}

//fills an array with values from offset counting upward
void fill_array(double**&A, int col, int row, double offset){
	for (int i = 0; i < row; i++){
		for (int j = 0; j < col; j++){
			A[i][j] = i*col + j + offset;
		}
	}
}

//fills an array with 0.0s
void fill_array_0s(double**&A, int col, int row){
	for (int i = 0; i < row; i++){
		for (int j = 0; j < col; j++){
			A[i][j] = 0.0;
		}
	}
}

// prints an output line to stdout in human readable form
void print_line(double*A, int size){
	for (int i = 0; i < size; i++){
		std::cout << A[i] << "\t";
	}
	std::cout << std::endl << std::endl;
}

// prints an array to stdout in human readable form
void print_array(double**A, int col, int row){
	for (int i = 0; i < row; i++){
		for (int j = 0; j < col; j++){
			std::cout << A[i][j] << "\t";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

//liniarizes an array of floats into a vector for processing
void vectorize_float_array(float**A, float*&a, int col, int row){
	for (int i = 0; i < row; i++){
		for (int j = 0; j < col; j++){
			a[i*col + j] = A[i][j];
		}
	}
}

//liniarizes an array into a vector for processing
void vectorize_array(double**A, double*&a, int col, int row){
	for (int i = 0; i < row; i++){
		for (int j = 0; j < col; j++){
			a[i*col + j] = A[i][j];
		}
	}
}

//converts a vectorized array back into a two dimensional format
void arrayize_vector(double*a, double**&A, int col, int row){
	for (int i = 0; i < row; i++){
		for (int j = 0; j < col; j++){
			A[i][j] = a[i*col + j];
		}
	}
}

//dummy function that adds two arrays and places them into a third
void simple_add(double**&C, double**A, double**B, int col, int row){
	for (int i = 0; i < row; i++){
		for (int j = 0; j < col; j++){
			C[i][j] = A[i][j] + B[i][j];
		}
	}
}

//returns a single value that is the average difference between two arrays ignoring a fixed number of rows/cols from the edge
double array_diff(double**A, double**B, int col, int row, int ignore){
	double running_error = 0.0;
	for (int i = ignore; i < (row - ignore); i++){
		for (int j = ignore; j < (col - ignore); j++){
			running_error += abs(A[i][j] - B[i][j]);
		}
	}
	return running_error / (double)((col - (2 * ignore))*(row - (2 * ignore)));
}

// returns a single value that is the maximum difference between two arrays ignoring a fixed number of rows/cols from the edge
double array_max_diff(double**A, double**B, int col, int row, int ignore){
	double max_error = 0.0;
	int maxi = 0;
	int maxj = 0;
	int counter = 0;
	for (int i = ignore; i < (row - ignore); i++){
		for (int j = ignore; j < (col - ignore); j++){
			if (abs(A[i][j] - B[i][j])>max_error){ max_error = abs(A[i][j] - B[i][j]); maxi = i; maxj = j; }
			if (abs(A[i][j] - B[i][j])> 15.0){ counter++;  std::cout << abs(A[i][j] - B[i][j]) << "\t" << i << "," << j << std::endl; }
			if (A[i][j] - B[i][j] != A[i][j] - B[i][j]){ std::cout << A[i][j] << "-" << B[i][j] << "=" << A[i][j] - B[i][j] << "\t" << i << "," << j << std::endl; }
		}
	}
	return max_error;
}

// prints the time difference between t1 and t2 in a human readable format as ms, s, or min.
void print_time(std::clock_t t2, std::clock_t t1){
	if (v){
		if ((t2 - t1 / (double)(CLOCKS_PER_SEC / 1000)) < 1000){
			std::cout << (t2 - t1) / (double)(CLOCKS_PER_SEC / 1000) << " ms\n";
		}
		else if (((t2 - t1) / (double)(CLOCKS_PER_SEC)) > 60){
			std::cout << (double)(t2 - t1) / (double)(CLOCKS_PER_SEC * 60) << " min\n";
		}
		else {
			std::cout << (t2 - t1) / (double)CLOCKS_PER_SEC << " s\n";
		}
	}
	bool a = true;
}

//cuts a set of edges from an array.  Deletes the input array and returns a cut down copy.
double** cut_once(double**& A, int& col, int& row){
	double**B;
	int i = 0;
	init_array(B, col - 2, row - 2);
	while (i < row - 2){
		int j = 0;
		while (j < col - 2){
			B[i][j] = A[i + 1][j + 1];
			j++;
		}
		i++;
	}
	deconstruct_array(A, row);
	A = NULL;
	col--;
	row--;
	col--;
	row--;
	return B;
}

// cuts n edges from an array
void cut_edges(double**& A, int& col, int& row, int n){
	int k = 0;
	while (k < n){
		A = cut_once(A, col, row);
		k++;
	}
}

// support function to be run device side which searches for a value in a sorted list and returns the index which is just above the value occuring
// contains sanitation for first and last value cases.
// it works in tandem with the other support functions though, I promise. Do not bounce.
__device__ int get_the_next_index(float*A, float b, int len){
	float min_val = 1.00e25;
	int index = 0;
	if (b < A[0]){ return 1; }
	if (b >= A[len - 1]){ return len - 1; }
	for (int i = 0; i < len; ++i){
		float diff = std::abs(A[i] - b);
		if (diff < min_val && (A[i] - b)>0){
			min_val = diff;
			index = i;
		}
		else if (diff < min_val && (A[i] - b) <= 0){
			min_val = diff;
			index = i + 1;
		}
	}
	return index;
}

//returns the address of the vectorized array when addressed in a two dimensional form.
__device__ int supportAddress(int i, int j, const int col){
	return i*col + j;
}


// references the table searching for the most apropriate alpha value
// weighted linear interpolation is used between the four closest alpha values.  when Sin and or Cin match exactly, the exact apropriate value is returned
// when Sin or Cin are >max or <min the max or min values are used respectively
__device__ double ref_table(double Sin, double Cin, float* Stab, float* Ctab, int Slen, int Clen, float* Atab){
	int x2 = get_the_next_index(Stab, Sin, Slen);
	int x1 = x2 - 1;
	double xlw, xhw;
	if (Sin >= Stab[Slen - 1]){
		xhw = 1.0;
	}
	else{
		xhw = (Sin - Stab[x1]) / (Stab[x2] - Stab[x1]);
	}
	xlw = 1.0 - xhw;
	int y2 = get_the_next_index(Ctab, Cin, Clen);
	int y1 = y2 - 1;
	double ylw, yhw;
	if (Cin >= Ctab[Clen - 1]){
		yhw = 1.0;
	}
	else{
		yhw = (Cin - Ctab[y1]) / (Ctab[y2] - Ctab[y1]);
	}
	ylw = 1.0 - yhw;
	int fir = supportAddress(y1, x1, Slen);
	int sec = supportAddress(y2, x1, Slen);
	int thi = supportAddress(y1, x2, Slen);
	int fo = supportAddress(y2, x2, Slen);
	double for_ret = (xlw*ylw)*Atab[fir] + (xlw*yhw)*Atab[sec] + (xhw*ylw)*Atab[thi] + (xhw*yhw)*Atab[fo];
	return for_ret;
}

// Here is where the magic happens
// device side kernel function which decouples
// iteratively goes through and finally outputs updated as the decoupled image in vectorized form.
__global__ void decoupleKernel(double *measured, double *updated, const int col, const int row, float*Stab, float* Ctab, float*Atab, int Slen, int Clen){
	int base = blockIdx.x*blockDim.x + threadIdx.x;
	int i = base / col;
	int j = base % col;
	int at = supportAddress(i, j, col);
	int above = supportAddress((i - 1), (j), col);
	int below = supportAddress((i + 1), (j), col);
	int left = supportAddress((i), (j - 1), col);
	int right = supportAddress((i), (j + 1), col);
	bool flag_1 = true;		//Flags are for edge sanitation
	bool flag_2 = true;
	bool flag_3 = true;
	bool flag_4 = true;
	float coef_1;			//alpha for each direction
	float coef_2;
	float coef_3;
	float coef_4;
	// A quick note, this implementation assumes that the coupling coefficient is symetric.
	//		That is to say that if 1% couples from a pixel to its left neighbor, 1% also coupled from the left neighbor to the pixel.
	//		I have no evidence to break that assumption but if broken, 4 additional coefficients can be added here to separate coupling coefficient in from out.
	if (i == 0){ coef_1 = 0.0; flag_1 = false; }
	else{ coef_1 = ref_table(updated[at], updated[above], Stab, Ctab, Slen, Clen, Atab); }
	if (i == (row - 1)){ coef_2 = 0.0; flag_2 = false; }
	else{ coef_2 = ref_table(updated[at], updated[below], Stab, Ctab, Slen, Clen, Atab); }
	if (j == 0){ coef_3 = 0.0; flag_3 = false; }
	else{ coef_3 = ref_table(updated[at], updated[left], Stab, Ctab, Slen, Clen, Atab); }
	if (j == (col - 1)){ coef_3 = 0.0; flag_4 = false; }
	else{ coef_4 = ref_table(updated[at], updated[right], Stab, Ctab, Slen, Clen, Atab); }
	double coup_in = 0.0;
	if (flag_1){ coup_in += coef_1*updated[above] * 0.01; }
	if (flag_2){ coup_in += coef_2*updated[below] * 0.01; }
	if (flag_3){ coup_in += coef_3*updated[left] * 0.01; }
	if (flag_4){ coup_in += coef_4*updated[right] * 0.01; }
	double coup_out = (coef_1 + coef_2 + coef_3 + coef_4)*0.01*updated[base];
	updated[base] = measured[base] - coup_in + coup_out;
}


// Helper function for using CUDA to decouple a vectorized array.
// for the decouple to work I need to know:
// the image array in vectorized form,
// the initial size of the array,
// decouple info (possibly as three arrays)
// deals with all the memory junk so I don't have to think about it again
// everything passed as arrays by reference or by value are deliberate and necessary
cudaError_t decWithCuda(double *&image, unsigned int col, unsigned int row, float*Stab, float*Ctab, float*Atab, unsigned int Slen, unsigned int Clen,
	cudaError_t cudaStatus, double *&dev_measured, double *&dev_updated, float *&dev_Stab, float *&dev_Ctab, float *&dev_Atab)
{
	unsigned int size = col*row;	// total length of the array

	cudaStatus = cudaMemcpy(dev_measured, image, size * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_updated, image, size * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_Ctab, Ctab, Clen * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_Stab, Stab, Slen * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_Atab, Atab, Clen * Slen * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU.
	int blocks;
	int threads;
	if (col == row){
		threads = col;
		blocks = row;
	}
	else if (col > row){
		threads = row;
		blocks = col;
	}
	else{
		threads = col;
		blocks = row;
	}
	int i = 0;
	while (i < 5){					//This is currently an iteration count, can be replaced with a convergance constraint
		decoupleKernel << <blocks, threads >> >(dev_measured, dev_updated, col, row, dev_Stab, dev_Ctab, dev_Atab, Slen, Clen);	// This is the line where decoupling is called
		i++;
	}
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(image, dev_updated, size * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

// yeah, I'm not wild about goto's either.
Error:
	cudaDeviceSynchronize();
	return cudaStatus;
}


// The main does the following in the following order:
//		A bunch of debug cycles to find memory leaks
//		Build coupling array from SCA file
//		Build a vector of file names to operate and save to from in_cou.txt and ou_cou.txt
//		For each image:
//			Turn image *.csv into an array
//			build an array to hold the output
//			For each iteration:
//				build subarrays that can fit into memory
//				vectorize subarrays
//				decouple column interface regions: populate output
//				decouple row interface regions: populate output
//				decouple box regions: populate output
//			Cycle 5 times (or until contraint is met)
//			Save decoupled array to output *.csv
//		Cycle for all images to be processed

int main()
{
	std::ofstream outfile;
	outfile.open("debug.txt");
	if (!outfile.is_open()){ return -1; }
	std::ifstream size_f;
	size_f.open("size.txt");
	if (!size_f.is_open()){ return -1; }
	int temsiz = 0;
	while (!size_f.eof()){
		size_f >> temsiz;
	}
	size_f.close();
	max_size = &temsiz;
	size_f.open("mem.txt");
	if (!size_f.is_open()){ return -1; }
	int temmem = 0;
	while (!size_f.eof()){
		size_f >> temmem;
	}
	int*mem_point = &temmem;
	int mem_max = *mem_point;
	std::cout << *max_size << std::endl;
	std::chrono::time_point<std::chrono::system_clock> stctime;
	stctime = std::chrono::system_clock::now();
	std::time_t stime = std::chrono::system_clock::to_time_t(stctime);
	std::cout << "Computation begins at " << std::ctime(&stime);
	outfile << "Computation begins at " << std::ctime(&stime);
	//Debug section used to check for memory leaks
	if (memory_check){
		int i;
		std::cout << "We are running a memory test here\n";
		std::cout << "Creation and clear of init_float_line\n";
		i = 0;
		while (i < 1000000){
			float*testA = NULL;
			init_float_line(testA, 512);
			deconstruct_float_line(testA);
			i++;
		}
		std::cout << "Press any key to advance to second test\n";
		std::cin.get();
		std::cout << "Creation and clear of init_line\n";
		i = 0;
		while (i < 1000000){
			double*testA = NULL;
			init_line(testA, 512);
			deconstruct_line(testA);
			i++;
		}
		std::cout << "Press any key to advance to third test\n";
		std::cin.get();
		std::cout << "Creation and clear of init_array\n";
		i = 0;
		while (i < 1000){
			double**testA = NULL;
			init_array(testA, 512, 512);
			deconstruct_array(testA, 512);
			i++;
		}
		std::cout << "Press any key to advance to the fourth test\n";
		std::cin.get();
		std::cout << "creation and clear of arrays inside from csv_to_array as tuples\n";
		i = 0;
		std::string fname = "C:\\Users\\Donlok\\AppData\\Local\\lxss\\home\\Donlok\\Documents\\Python\\Gen\\CSV_BRI\\Start_0_0\\mock_fixed_0ratio_1offset_0RMS.csv";
		while (i < 1000){
			std::tuple<double**, int, int> testCSV;
			testCSV = csv_to_array(fname, 0, 128, 0, 128);
			deconstruct_tuple(testCSV);
			i++;
		}
		std::cout << "Press any key to advance to the fifth test\n";
		std::cin.get();
		std::cout << "creation and clear of subarrays\n";
		i = 0;
		double **testA = NULL;
		init_array(testA, 512, 512);
		while (i < 10000){
			SubArray testhat;
			array_to_subarray(testA, testhat, 10, 20, 10, 20);
			deconstruct_subarray(testhat);
			i++;
		}
		std::cout << "Press any key to complete memory check\n";
		std::cin.get();
		std::cout << "Memory test complete\n";
	}
	//End of debug section testing the memory behavior of all deconstructors.

	if (v){ std::cout << "Begining initialization:\t\t\t"; }
	std::clock_t timeSTART = std::clock();
	std::tuple<float**, float*, float*, int, int> ASC_Ssize_Csize;
	float** A;
	float* S;
	float* C;
	int Ssize;
	int Csize;
	std::clock_t timeALT1 = std::clock();
	print_time(timeALT1, timeSTART);
	if (v){ std::cout << "Building look-up table:\t\t\t\t"; }
	//This is the file that contains the look-up table formatted in a particular way.  See readme.  At the moment it is generated by a python script.
	ASC_Ssize_Csize = build_table("look_up_true_large.txt");
	A = std::get<0>(ASC_Ssize_Csize);
	S = std::get<1>(ASC_Ssize_Csize);
	C = std::get<2>(ASC_Ssize_Csize);
	Ssize = std::get<3>(ASC_Ssize_Csize);
	Csize = std::get<4>(ASC_Ssize_Csize);
	std::clock_t timeALT0 = std::clock();
	print_time(timeALT0, timeALT1);
	if (v){ std::cout << "Building csv array:\t\t\t\t\n"; }
	std::tuple<double**, int, int> CSV;

	// Alright, so here I am going to include a section which fills 2 vectors of strings with filenames to be read later and processed as inputs and outputs
	// These files must have the same number of lines and the lines must contain viable file names of *.csv files to be used as input or outputs
	std::vector<std::string> in_cou;
	std::vector<std::string> ou_cou;

	std::string line;

	// These files contain a list of file names for input *.csv and locations to save output *.csv
	std::ifstream inf_cou("in_cou.txt");
	std::ifstream ouf_cou("ou_cou.txt");


	if (inf_cou.is_open()){
		while (std::getline(inf_cou, line)){
			in_cou.push_back(line);
		}
		inf_cou.close();
		while (std::getline(ouf_cou, line)){
			ou_cou.push_back(line);
		}
		ouf_cou.close();
	}

	// Buffer allocation here keeps GPU memory stable

	double *dev_measured = NULL;		// device vector storing the input image to be used in calculations
	double *dev_updated = NULL;			// device vector to update and ultimately contain the restored image
	unsigned int size;					// total length of the array
	if (*max_size < 512){ size = (*max_size)*(*max_size); }
	else{ size = 512 * 512; }
	float *dev_Stab = NULL;
	float *dev_Ctab = NULL;
	float *dev_Atab = NULL;
	cudaError_t cudaStatus;
	cudaStatus = cudaDeviceReset();
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
	cudaStatus = cudaMalloc((void**)&dev_measured, size * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&dev_updated, size * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	cudaStatus = cudaMalloc((void**)&dev_Stab, Ssize * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&dev_Ctab, Csize * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&dev_Atab, Csize * Ssize * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	int term_cycle = 100000;
	int cycle = 0;
	double** Q = NULL;
	while (!in_cou.empty() && (!(cycle == term_cycle))){
		cycle++;

		std::string input1 = in_cou.back();
		in_cou.pop_back();
		std::string output1 = ou_cou.back();
		ou_cou.pop_back();
		if (v){ std::cout << "Array load" << std::endl; }
		CSV = csv_to_array(input1, 0, *max_size, 0, *max_size);
		if (v){ std::cout << input1 << std::endl; }
		if (CSV == std::make_tuple(Q, 0, 0)){ return -10; }	// This indicates that the file failed to load

		double**csv = NULL;
		int ccol;
		int crow;
		int tcol;
		int trow;

		csv = std::get<0>(CSV);
		ccol = std::get<1>(CSV);
		crow = std::get<2>(CSV);

		float*vecA = NULL;
		double*vecCSV = NULL;
		double** result = NULL;
		int boxes = 0;

		init_array(result, ccol, crow);
		fill_array_0s(result, ccol, crow);

		init_float_line(vecA, Ssize*Csize);
		vectorize_float_array(A, vecA, Ssize, Csize);
		timeALT1 = std::clock();
		int cut = 10;
		//When this loop exits, the image has been decoupled
		if (*max_size > mem_max){		//only enter this sub loop if the image has to be subdivided.
			//Calculation for the vertical boxes
			for (int p = mem_max-32; (p + 64) < ccol; p = p + mem_max){
				int q = 0;
				SubArray hat;
				array_to_subarray(csv, hat, p, q, 64, crow);
				init_line(vecCSV, hat.rows*hat.cols);
				vectorize_array(hat.values, vecCSV, hat.cols, hat.rows);
				if (v){ std::cout << "Begining decouple:\t\t\t\t"; }
				timeALT0 = std::clock();
				cudaStatus = decWithCuda(vecCSV, hat.cols, hat.rows, S, C, vecA, Ssize, Csize, cudaStatus, dev_measured, dev_updated, dev_Stab, dev_Ctab, dev_Atab);
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "decWithCuda failed!");
					return 1;
				}
				boxes++;
				timeALT1 = std::clock();
				print_time(timeALT1, timeALT0);
				arrayize_vector(vecCSV, hat.values, hat.cols, hat.rows);
				cut_edges(hat.values, hat.cols, hat.rows, cut);
				hat.startcol = hat.startcol + cut;
				hat.startrow = hat.startrow + cut;
				subarray_into_array(result, hat, crow, ccol);
				deconstruct_subarray(hat);
				deconstruct_line(vecCSV);
			}
			// calculations of the horizontal boxes
			for (int q = mem_max-32; (q + 64) < crow; q = q + mem_max){
				int p = 0;
				SubArray hat;
				array_to_subarray(csv, hat, p, q, ccol, 64);
				init_line(vecCSV, hat.rows*hat.cols);
				vectorize_array(hat.values, vecCSV, hat.cols, hat.rows);
				if (v){ std::cout << "Begining decouple:\t\t\t\t"; }
				timeALT0 = std::clock();
				cudaStatus = decWithCuda(vecCSV, hat.cols, hat.rows, S, C, vecA, Ssize, Csize, cudaStatus, dev_measured, dev_updated, dev_Stab, dev_Ctab, dev_Atab);
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "decWithCuda failed!");
					return 1;
				}
				boxes++;
				timeALT1 = std::clock();
				print_time(timeALT1, timeALT0);
				arrayize_vector(vecCSV, hat.values, hat.cols, hat.rows);
				cut_edges(hat.values, hat.cols, hat.rows, cut);
				hat.startcol = hat.startcol + cut;
				hat.startrow = hat.startrow + cut;
				subarray_into_array(result, hat, crow, ccol);
				deconstruct_subarray(hat);
				deconstruct_line(vecCSV);
			}
		}
		//calculation of large boxes
		int max_size_int;
		if (*max_size > mem_max){ max_size_int = mem_max; }
		else{ max_size_int = *max_size; }
		if (v){ std::cout << "Begining box decouple" << std::endl; }
		for (int p = 0; p + max_size_int <= crow; p = p + max_size_int){
			for (int q = 0; q + max_size_int <= ccol; q = q + max_size_int){
				SubArray hat;
				array_to_subarray(csv, hat, q, p, max_size_int, max_size_int);
				init_line(vecCSV, hat.rows*hat.cols);
				vectorize_array(hat.values, vecCSV, hat.cols, hat.rows);
				std::cout << "Begining decouple:\t\t\t\t" << cycle << "\n";
				timeALT0 = std::clock();
				cudaStatus = decWithCuda(vecCSV, max_size_int, max_size_int, S, C, vecA, Ssize, Csize, cudaStatus, dev_measured, dev_updated, dev_Stab, dev_Ctab, dev_Atab);
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "decWithCuda failed!");
					return 1;
				}
				boxes++;
				timeALT1 = std::clock();
				print_time(timeALT1, timeALT0);
				arrayize_vector(vecCSV, hat.values, hat.cols, hat.rows);
				cut_edges(hat.values, hat.cols, hat.rows, (cut));
				hat.startcol = hat.startcol + (cut);
				hat.startrow = hat.startrow + (cut);
				subarray_into_array(result, hat, crow, ccol);
				deconstruct_subarray(hat);
				deconstruct_line(vecCSV);
			}
		}
		timeALT0 = std::clock();
		std::clock_t time2 = std::clock();
		if (v){ std::cout << input1 << std::endl; }
		if (v){ std::cout << "With Cuda, dectime of: \t\t\t\t"; }
		print_time(time2, timeSTART);
		array_to_csv(result, ccol, crow, output1);
		deconstruct_float_line(vecA);
		deconstruct_array(result, crow);
		deconstruct_tuple(CSV);
	}
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	outfile << cycle << " files decoupled" << std::endl;
	std::chrono::time_point<std::chrono::system_clock> etctime;
	etctime = std::chrono::system_clock::now();
	std::time_t etime = std::chrono::system_clock::to_time_t(etctime);
	outfile << "Computation ends at " << std::ctime(&etime);
	outfile.close();
	return 1337;
}
// this outputs .csv files with naming convention established by the file ./output.txt
// res.csv is a .csv file containing the IPC corrected output image
// currently res.csv and diff.csv both contain an edge of invalid pixels.  The ten pixels nearest to the outer edge are invalid.
// This is because the decoupling algorithm is currently set to operate on each subarray ten times.
// Error propogates one pixel in from the outer edge on each iteration.
// 10 is likely overkill as the algorithim generally converges after 3-5 iterations.
// Last updated March 12, 2018
// The latest update fixes all observed memory leaks and introduces unique deconstructors for each custom array, vector, subarray, and tuple types.
// It is worth noting that to properly free the memory occupied by an instance of the loaded tuple objects either deconstruct_tuple can be called on the tuple or deconstruct_array can be called on the first facet of the tuple.
// That is to say for a tuple TUPLE deconstruction can be achieved through either deconstruct_tuple(TUPLE); or deconstruct_array(std::get<0>(TUPLE));  but NOT both.
// At this point the size is set to autoconfigure with the constraint that the initial constant int max_size be declared to be the array size.  Also only square arrays.