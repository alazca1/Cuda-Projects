#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <windows.h>
#include <stdlib.h>
#include <string.h>
#include <string>


// Image size
#define WIDTH          640
#define HEIGHT         480
#define IMAGE_SIZE     (WIDTH * HEIGHT)
#define IMAGE_SIZE_RGB (IMAGE_SIZE * 3)

// Types definitions
typedef unsigned char byte; // Number in range [0-255]


///////////////////////////////////////////////////////////////////////////////////////////////////
// Utilities
///////////////////////////////////////////////////////////////////////////////////////////////////

struct BMPHeader
{
	char bfType[2];       /* "BM" */
	int bfSize;           /* Size of file in bytes */
	int bfReserved;       /* set to 0 */
	int bfOffBits;        /* Byte offset to actual bitmap data (= 54) */
	int biSize;           /* Size of BITMAPINFOHEADER, in bytes (= 40) */
	int biWidth;          /* Width of image, in pixels */
	int biHeight;         /* Height of images, in pixels */
	short biPlanes;       /* Number of planes in target device (set to 1) */
	short biBitCount;     /* Bits per pixel (24 in this case) */
	int biCompression;    /* Type of compression (0 if no compression) */
	int biSizeImage;      /* Image size, in bytes (0 if no compression) */
	int biXPelsPerMeter;  /* Resolution in pixels/meter of display device */
	int biYPelsPerMeter;  /* Resolution in pixels/meter of display device */
	int biClrUsed;        /* Number of colors in the color table (if 0, use maximum allowed by biBitCount) */
	int biClrImportant;   /* Number of important colors.  If 0, all colors are important */
};

int write_bmp(const char* filename, int width, int height, char* rgb)
{
	int i, j, ipos;
	int bytesPerLine;
	char* line;

	FILE* file;
	struct BMPHeader bmph;

	/* The length of each line must be a multiple of 4 bytes */

	bytesPerLine = (3 * (width + 1) / 4) * 4;

	strcpy(bmph.bfType, "BM");
	bmph.bfOffBits = 54;
	bmph.bfSize = bmph.bfOffBits + bytesPerLine * height;
	bmph.bfReserved = 0;
	bmph.biSize = 40;
	bmph.biWidth = width;
	bmph.biHeight = height;
	bmph.biPlanes = 1;
	bmph.biBitCount = 24;
	bmph.biCompression = 0;
	bmph.biSizeImage = bytesPerLine * height;
	bmph.biXPelsPerMeter = 0;
	bmph.biYPelsPerMeter = 0;
	bmph.biClrUsed = 0;
	bmph.biClrImportant = 0;

	file = fopen(filename, "wb");
	if (file == NULL)
		return 0;

	fwrite(&bmph.bfType, 2, 1, file);
	fwrite(&bmph.bfSize, 4, 1, file);
	fwrite(&bmph.bfReserved, 4, 1, file);
	fwrite(&bmph.bfOffBits, 4, 1, file);
	fwrite(&bmph.biSize, 4, 1, file);
	fwrite(&bmph.biWidth, 4, 1, file);
	fwrite(&bmph.biHeight, 4, 1, file);
	fwrite(&bmph.biPlanes, 2, 1, file);
	fwrite(&bmph.biBitCount, 2, 1, file);
	fwrite(&bmph.biCompression, 4, 1, file);
	fwrite(&bmph.biSizeImage, 4, 1, file);
	fwrite(&bmph.biXPelsPerMeter, 4, 1, file);
	fwrite(&bmph.biYPelsPerMeter, 4, 1, file);
	fwrite(&bmph.biClrUsed, 4, 1, file);
	fwrite(&bmph.biClrImportant, 4, 1, file);

	line = (char*)malloc(bytesPerLine);

	for (i = height - 1; i >= 0; i--)
	{
		for (j = 0; j < width; j++)
		{
			ipos = 3 * (width * i + j);
			line[3 * j] = rgb[ipos + 2];
			line[3 * j + 1] = rgb[ipos + 1];
			line[3 * j + 2] = rgb[ipos];
		}
		fwrite(line, bytesPerLine, 1, file);
	}

	free(line);
	fclose(file);

	return 1;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// Gray-Scale image (CPU)
///////////////////////////////////////////////////////////////////////////////////////////////////

void sample_01_CPU()
{
	const char* filename = "imagen_001_CPU.bmp";

	byte* rgb = new byte[IMAGE_SIZE_RGB];

	int ipos;
	for (int j = 0; j < HEIGHT; j++)
	{
		for (int i = 0; i < WIDTH; i++)
		{
			int Gray = (i + j) * 255 / (WIDTH + HEIGHT);
			ipos = 3 * (WIDTH * j + i);

			rgb[ipos] = Gray;   //R
			rgb[ipos + 2] = Gray;   //B
			rgb[ipos + 1] = Gray;	//G			
		}
	}

	write_bmp(filename, WIDTH, HEIGHT, (char*)rgb);
	delete[] rgb;
	printf("sample_01_CPU done... --> %s \n", filename);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// Gray & Color Scale images (GPU)
///////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void gray_scale_kernel(byte* rgb, int w, int h)
{
	int x = threadIdx.x + (blockIdx.x * blockDim.x);
	int y = threadIdx.y + (blockIdx.y * blockDim.y);
	byte gray = (x + y) * 255 / (w + h);
	int i = 3 * ((w * y) + x);

	rgb[i] = gray;
	rgb[i + 1] = gray;
	rgb[i + 2] = gray;
}

__global__ void color_scale_kernel(byte* rgb, int w, int h)
{
	int x = threadIdx.x + (blockIdx.x * blockDim.x);
	int y = threadIdx.y + (blockIdx.y * blockDim.y);
	byte r = (x + y) * 255 / (w + h);
	byte g = 100;
	byte b = (x + y) * 255 / (w + h);
	int i = 3 * ((w * y) + x);

	rgb[i] = r;
	rgb[i + 1] = g;
	rgb[i + 2] = b;
}

void sample_01_GPU()
{
	const char* filename01 = "imagen_001_GPU.bmp";
	const char* filename02 = "imagen_002_GPU.bmp";
	const int THREADS_PER_BLOCK_X = 16;
	const int THREADS_PER_BLOCK_Y = 32;
	const int NUM_BLOCK_X = 40;
	const int NUM_BLOCK_Y = 15;

	dim3 block_size(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1);
	dim3 grid_size(NUM_BLOCK_X, NUM_BLOCK_Y, 1);

	// Allocate CPU memory
	byte* rgb = new byte[IMAGE_SIZE_RGB];

	// GPU pointer
	byte* dev_rgb = 0;

	// Allocate GPU memory
	cudaMalloc((void**)&dev_rgb, IMAGE_SIZE_RGB * sizeof(byte));

	// Gray scale sample ///////////////////////////////////////////////////////////

	// Get the color for each pixel
	gray_scale_kernel <<< grid_size, block_size >>> (dev_rgb, WIDTH, HEIGHT);

	// Copy the memory to CPU
	cudaMemcpy(rgb, dev_rgb, IMAGE_SIZE_RGB * sizeof(byte), cudaMemcpyDeviceToHost);

	// Write bmp file
	write_bmp(filename01, WIDTH, HEIGHT, (char*)rgb);

	// Some debug
	printf("sample_01_GPU gray-scale done... --> %s \n", filename01);
	//////////////////////////////////////////////////////////////////////////////


	// Color scale sample ///////////////////////////////////////////////////////////

	// Get the color for each pixel
	color_scale_kernel <<< grid_size, block_size >>> (dev_rgb, WIDTH, HEIGHT);

	// Copy the memory to CPU
	cudaMemcpy(rgb, dev_rgb, IMAGE_SIZE_RGB * sizeof(byte), cudaMemcpyDeviceToHost);

	// Write bmp file
	write_bmp(filename02, WIDTH, HEIGHT, (char*)rgb);

	// Some debug
	printf("sample_01_GPU color-scale done... --> %s \n", filename02);
	//////////////////////////////////////////////////////////////////////////////


	// Free GPU memory
	cudaFree(dev_rgb);
	delete[] rgb;
}



///////////////////////////////////////////////////////////////////////////////////////////////////
// Mandelbrot Sample
///////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void mandelbrot_kernel(byte* rgb, int W, int H, int iteraciones, int limite)
{
	int x = threadIdx.x + (blockIdx.x * blockDim.x);
	int y = threadIdx.y + (blockIdx.y * blockDim.y);
	int i = 3 * ((W * y) + x);
	int gray;
	int r_g_hot_threshold = 90;
	int g_b_hot_threshold = 180;
	int r_coeff = 255 / r_g_hot_threshold;
	int g_coeff = 255 / (g_b_hot_threshold - r_g_hot_threshold);
	int b_coeff = 255 / (255 - g_b_hot_threshold);


	// Funcion Mandelbrot
	float z[2] = { 0, 0 };
	float c[2] = { (x * 1.f) / W, (y * 1.f) / H};

	for (int ij = 0; ij < iteraciones; ij++)
	{
		float w[2];
		float az;
		w[0] = (z[0] * z[0]) - (z[1] * z[1]);
		w[1] = 2 * z[0] * z[1];
		z[0] = w[0] + c[0];
		z[1] = w[1] + c[1];
		az = sqrtf((z[0] * z[0]) + (z[1] * z[1]));

		if (az >= limite) {
			gray = ij;
			break;
		}
		else {
			gray = ij;
		}
	}

	byte gray_rgb = byte(gray * 255 / iteraciones);

	rgb[i] = gray_rgb < r_g_hot_threshold ? (gray_rgb * r_coeff) - 255 : 255;
	rgb[i + 1] = gray_rgb < r_g_hot_threshold ? 0 : (gray_rgb > g_b_hot_threshold ? 255 : (gray_rgb * g_coeff) - 255);
	rgb[i + 2] = gray_rgb < g_b_hot_threshold ? 0 : (gray_rgb * b_coeff) - 255;
}

void sample_02_GPU_mandelbrot()
{
	// Output filename	
	const char* filename_prefix = "imagen_002_GPU_mandel";

	// Constants
	const int w = WIDTH;
	const int h = HEIGHT;
	const int THREADS_PER_BLOCK_X = 16;
	const int THREADS_PER_BLOCK_Y = 32;
	const int NUM_BLOCK_X = 40;
	const int NUM_BLOCK_Y = 15;

	dim3 block_size(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1);
	dim3 grid_size(NUM_BLOCK_X, NUM_BLOCK_Y, 1);

	// Allocate CPU memory
	byte* rgb = new byte[IMAGE_SIZE_RGB];

	// GPU pointer
	byte* dev_rgb;

	// Allocate GPU memory
	cudaMalloc((void**)&dev_rgb, IMAGE_SIZE_RGB * sizeof(byte));


	// Mandelbrot /////////////////////////////////////////////////////////////////

	// Mandelbrot params
	int iteraciones = 20;
	int limite = 2;

	// Get the color for each pixel	
	mandelbrot_kernel <<< grid_size, block_size >>> (dev_rgb, w, h, iteraciones, limite);

	// Copy the memory to CPU
	cudaMemcpy(rgb, dev_rgb, IMAGE_SIZE_RGB * sizeof(byte), cudaMemcpyDeviceToHost);

	// Write bmp file
	std::string s(filename_prefix);
	s.append(".bmp");;
	write_bmp(s.c_str(), w, h, (char*)rgb);

	// Some debug
	printf("sample_01_GPU mandelbrot done... --> %s \n", s.c_str());


	//////////////////////////////////////////////////////////////////////////////

	// Free GPU memory
	cudaFree(dev_rgb);
	delete[] rgb;

	// Free CPU memory
	// ...
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// Thresholding Samples
///////////////////////////////////////////////////////////////////////////////////////////////////


__global__ void threshold_kernel(byte* rgb, int W, int H, byte th)
{
	int x = threadIdx.x + (blockIdx.x * blockDim.x);
	int y = threadIdx.y + (blockIdx.y * blockDim.y);
	int i = 3 * ((W * y) + x);

	rgb[i] = rgb[i] > th ? 255 : 0;
	rgb[i + 1] = rgb[i + 1] > th ? 255 : 0;
	rgb[i + 2] = rgb[i + 2] > th ? 255 : 0;
}

__global__ void threshold_kernel(byte* rgb, byte* rgb_out, int W, int H, byte th)
{
	int x = threadIdx.x + (blockIdx.x * blockDim.x);
	int y = threadIdx.y + (blockIdx.y * blockDim.y);
	int i = 3 * ((W * y) + x);

	rgb_out[i] = rgb[i] > th ? 255 : 0;
	rgb_out[i + 1] = rgb[i + 1] > th ? 255 : 0;
	rgb_out[i + 2] = rgb[i + 2] > th ? 255 : 0;
}

__global__ void threshold_local_kernel(byte* rgb, int W, int H, byte th1, byte th2)
{
	int x = threadIdx.x + (blockIdx.x * blockDim.x);
	int y = threadIdx.y + (blockIdx.y * blockDim.y);
	int i = 3 * ((W * y) + x);

	float ramp_coefficient = 255/(th2 - th1);
	byte r = byte((rgb[i] * ramp_coefficient) - 255);
	byte g = byte((rgb[i + 1] * ramp_coefficient) - 255);
	byte b = byte((rgb[i + 2] * ramp_coefficient) - 255);

	rgb[i] = rgb[i] < th1 ? 0 : (rgb[i] > th2 ? 255 : r);
	rgb[i + 1] = rgb[i + 1] < th1 ? 0 : (rgb[i + 1] > th2 ? 255 : g);
	rgb[i + 2] = rgb[i + 2] < th1 ? 0 : (rgb[i + 2] > th2 ? 255 : b);
}

void sample_03_GPU_thresholding()
{
	// Output filename	
	const char* filename01 = "imagen_003_GPU_th.bmp";
	const char* filename02 = "imagen_003_GPU_local_th.bmp";
	const int THREADS_PER_BLOCK_X = 16;
	const int THREADS_PER_BLOCK_Y = 32;
	const int NUM_BLOCK_X = 40;
	const int NUM_BLOCK_Y = 15;

	dim3 block_size(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1);
	dim3 grid_size(NUM_BLOCK_X, NUM_BLOCK_Y, 1);

	// Constants
	const int w = WIDTH;
	const int h = HEIGHT;

	// Allocate CPU memory
	byte* rgb = new byte[IMAGE_SIZE_RGB];

	// GPU pointer
	byte* dev_rgb;

	// Allocate GPU memory
	cudaMalloc((void**)&dev_rgb, IMAGE_SIZE_RGB * sizeof(byte));

	// Thresholding /////////////////////////////////////////////////////////////////

	// Get the color for each pixel
	gray_scale_kernel <<<grid_size, block_size >>>(dev_rgb, w, h);

	// Apply the thresholding
	byte th = 100;
	threshold_kernel <<<grid_size, block_size >>>(dev_rgb, w, h, th);

	// Copy the memory to CPU
	cudaMemcpy(rgb, dev_rgb, IMAGE_SIZE_RGB * sizeof(byte), cudaMemcpyDeviceToHost);

	// Write bmp file
	write_bmp(filename01, w, h, (char*)rgb);

	// Some debug
	printf("sample_03_GPU thresholding done... --> %s \n", filename01);


	// Local Thresholding ////////////////////////////////////////////////////////////

	// Get the color for each pixel
	gray_scale_kernel <<<grid_size, block_size >>> (dev_rgb, w, h);

	// Apply the thresholding
	byte th1 = 100;
	byte th2 = 200;
	threshold_local_kernel <<<grid_size, block_size >>> (dev_rgb, w, h, th1, th2);

	// Copy the memory to CPU
	cudaMemcpy(rgb, dev_rgb, IMAGE_SIZE_RGB * sizeof(byte), cudaMemcpyDeviceToHost);

	// Write bmp file
	write_bmp(filename02, w, h, (char*)rgb);

	// Some debug
	printf("sample_03_GPU thresholding done... --> %s \n", filename02);

	/////////////////////////////////////////////////////////////////////////////////

	// Free GPU memory
	cudaFree(dev_rgb);
	delete[] rgb;
}



///////////////////////////////////////////////////////////////////////////////////////////////////
// Color Scale images + Morphological operations (GPU)
///////////////////////////////////////////////////////////////////////////////////////////////////

// (1) - Color scale sample /////////////////////////////////////////////////////////

__global__ void color_scale_kernel_sesion02(byte* rgb, int w, int h)
{
	int posx = blockIdx.x;
	int posy = threadIdx.x;

	rgb[(posx + (posy * w)) * 3 + 0] = posy;  //R
	rgb[(posx + (posy * w)) * 3 + 1] = posx;  //G
	rgb[(posx + (posy * w)) * 3 + 2] = 100;   //B
}


// (2) Translate sample /////////////////////////////////////////////////////////////

__device__ void matrix_vector_mult(float* matrix, float* vector, float* vector_out)
{
	vector_out[0] = matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2];
	vector_out[1] = matrix[3] * vector[0] + matrix[4] * vector[1] + matrix[5] * vector[2];
	vector_out[2] = matrix[6] * vector[0] + matrix[7] * vector[1] + matrix[8] * vector[2];
}

__global__ void traslation_kernel(byte* rgb, byte* rgb_out, int w, int h)
{
	// Transformation parameters
	float t_x = -200.0f;
	float t_y = 100.0f;

	// Region to work with
	float x_min = 450.0f;
	float x_max = 550.0f;
	float y_min = 200.0f;
	float y_max = 300.0f;

	// Position for the pixel
	int posx = blockIdx.x;
	int posy = threadIdx.x;

	// Always copy the original image to the source
	rgb_out[(posx + posy * w) * 3 + 0] = rgb[(posx + posy * w) * 3 + 0];    //R
	rgb_out[(posx + posy * w) * 3 + 1] = rgb[(posx + posy * w) * 3 + 1];	//G
	rgb_out[(posx + posy * w) * 3 + 2] = rgb[(posx + posy * w) * 3 + 2];	//B

	// Wait to copy all the image values for this block
	__syncthreads();

	// If we are in the region of interest...
	if (posx >= x_min && posx <= x_max &&
		posy >= y_min && posy <= y_max)
	{
		// Transformation matrix
		float matrix_tx[9] = { 1, 0, t_x, 0, 1, t_y, 0, 0, 1 };

		// Initial position
		float vector[3] = { posx, posy, 1 };

		// Final position
		float vec_out[3];

		// Get the final position
		matrix_vector_mult(matrix_tx, vector, vec_out);
		int posx_out = vec_out[0];
		int posy_out = vec_out[1];

		// Set the output image values
		rgb_out[(posx_out + posy_out * w) * 3 + 0] = rgb[(posx + posy * w) * 3 + 0];    //R
		rgb_out[(posx_out + posy_out * w) * 3 + 1] = rgb[(posx + posy * w) * 3 + 1];	//G
		rgb_out[(posx_out + posy_out * w) * 3 + 2] = rgb[(posx + posy * w) * 3 + 2];	//B
	}
}


// (3) Translate & Rotate  //////////////////////////////////////////////////////////

__device__ float deg2rad(float angle_deg)
{
	return angle_deg * 3.1416f / 180;
}

__global__ void traslation_rotation_kernel(byte* rgb, byte* rgb_out, int w, int h)
{
	// Region to work with
	float x_min = 450.0f;
	float x_max = 550.0f;
	float y_min = 200.0f;
	float y_max = 300.0f;

	// Transformation parameters
	float t_x = -x_min - ((x_max - x_min) / 2);
	float t_y = -y_min - ((y_max - y_min) / 2);

	// Rotation parameters
	float rad = deg2rad(45.f);

	// Position for the pixel
	int posx = blockIdx.x;
	int posy = threadIdx.x;

	// Always copy the original image to the source
	rgb_out[(posx + posy * w) * 3 + 0] = rgb[(posx + posy * w) * 3 + 0];    //R
	rgb_out[(posx + posy * w) * 3 + 1] = rgb[(posx + posy * w) * 3 + 1];	//G
	rgb_out[(posx + posy * w) * 3 + 2] = rgb[(posx + posy * w) * 3 + 2];	//B

	// Wait to copy all the image values for this block
	__syncthreads();

	// If we are in the region of interest...
	if (posx >= x_min && posx <= x_max &&
		posy >= y_min && posy <= y_max)
	{
		// Transformation matrix
		float t_x2 = -200.f;
		float t_y2 = 100.f;
		float matrix_tx_tras1[9] = { 1, 0, t_x, 0, 1, t_y, 0, 0, 1 };
		float matrix_tx_rot[9] = { cos(rad), -sin(rad), 0, sin(rad), cos(rad), 0, 0, 0, 1 };
		float matrix_tx_tras2[9] = { cos(rad)/2, -sin(rad), posx + t_x2, sin(rad), cos(rad)/2, posy + t_y2, 0, 0, 1 };

		// Initial position
		float vector[3] = { posx, posy, 1 };

		// Final position
		float vec_out_tras1[3];
		float vec_out_tras2[3];
		float vec_out_rot[3];
		
		// Get the final position
		matrix_vector_mult(matrix_tx_tras1, vector, vec_out_tras1);
		matrix_vector_mult(matrix_tx_rot, vec_out_tras1, vec_out_rot);
		matrix_vector_mult(matrix_tx_tras2, vec_out_rot, vec_out_tras2);

		int posx_out = int(vec_out_tras2[0]);
		int posy_out = int(vec_out_tras2[1]);

		// Set the output image values
		rgb_out[(posx_out + posy_out * w) * 3 + 0] = rgb[(posx + posy * w) * 3 + 0];    //R
		rgb_out[(posx_out + posy_out * w) * 3 + 1] = rgb[(posx + posy * w) * 3 + 1];	//G
		rgb_out[(posx_out + posy_out * w) * 3 + 2] = rgb[(posx + posy * w) * 3 + 2];	//B
	}
}


// (4) Translate & Rotate (Advanced version) ////////////////////////////////////////

__device__ void matrix_3x3_mult(float* m1, float* m2, float* m_out)
{
	// ... STUDENT CODE
}

__global__ void traslation_rotation_kernel_advanced01(byte* rgb, byte* rgb_out, int w, int h)
{
	// ... STUDENT CODE
}


// (5) Translate & Rotate (Advanced version-2) //////////////////////////////////////

__global__ void calculate_traslation_rotation_kernel(float* m_out)
{
	// ... STUDENT CODE
}

__global__ void traslation_rotation_kernel_advanced02(byte* rgb, byte* rgb_out, float* m, int w, int h)
{
	// ... STUDENT CODE
}


// -------- kERNEL FUNCTIONS CALLER...

void sample_03_GPU_geom_tx()
{
	const char* filename01 = "imagen_s02_color_scale_GPU.bmp";
	const char* filename02 = "imagen_s02_color_scale_GPU_tr.bmp";
	const char* filename03 = "imagen_s02_color_scale_GPU_tr_rt.bmp";
	const char* filename04 = "imagen_s02_color_scale_GPU_tr_rt_advanced01.bmp";
	const char* filename05 = "imagen_s02_color_scale_GPU_tr_rt_advanced02.bmp";

	// Allocate CPU memory
	byte* rgb = new byte[IMAGE_SIZE_RGB];

	// GPU pointers
	byte* dev_rgb01;
	byte* dev_rgb02;

	// Allocate GPU memory
	cudaMalloc((void**)&dev_rgb01, IMAGE_SIZE_RGB * sizeof(byte));
	cudaMalloc((void**)&dev_rgb02, IMAGE_SIZE_RGB * sizeof(byte));


	// (1) - Color scale sample /////////////////////////////////////////////////////////

	// Get the color for each pixel
	color_scale_kernel_sesion02 <<< WIDTH, HEIGHT >>> (dev_rgb01, WIDTH, HEIGHT);

	// Copy the memory to CPU
	cudaMemcpy(rgb, dev_rgb01, IMAGE_SIZE_RGB * sizeof(byte), cudaMemcpyDeviceToHost);

	// Write bmp file
	write_bmp(filename01, WIDTH, HEIGHT, (char*)rgb);

	// Some debug
	printf("GPU color-scale done... --> %s \n", filename01);
	/////////////////////////////////////////////////////////////////////////////////////


	// (2) Translate sample /////////////////////////////////////////////////////////////

	// Get the color for each pixel
	traslation_kernel <<< WIDTH, HEIGHT >>> (dev_rgb01, dev_rgb02, WIDTH, HEIGHT);

	// Copy the memory to CPU
	cudaMemcpy(rgb, dev_rgb02, IMAGE_SIZE_RGB * sizeof(byte), cudaMemcpyDeviceToHost);

	// Write bmp file
	write_bmp(filename02, WIDTH, HEIGHT, (char*)rgb);

	// Some debug
	printf("GPU color-scale translate done... --> %s \n", filename02);
	/////////////////////////////////////////////////////////////////////////////////////


	// (3) Translate & Rotate  //////////////////////////////////////////////////////////

	// Get the color for each pixel
	traslation_rotation_kernel <<< WIDTH, HEIGHT >>> (dev_rgb01, dev_rgb02, WIDTH, HEIGHT);

	// Copy the memory to CPU
	cudaMemcpy(rgb, dev_rgb02, IMAGE_SIZE_RGB * sizeof(byte), cudaMemcpyDeviceToHost);

	// Write bmp file
	write_bmp(filename03, WIDTH, HEIGHT, (char*)rgb);

	// Some debug
	printf("GPU color-scale translate + rotate done... --> %s \n", filename03);
	/////////////////////////////////////////////////////////////////////////////////////


	// (4) Translate & Rotate (Advanced version) ////////////////////////////////////////

	// Get the color for each pixel
	// traslation_rotation_kernel_advanced01 <<< WIDTH, HEIGHT >>> (dev_rgb01, dev_rgb02, WIDTH, HEIGHT);

	// Copy the memory to CPU
	// ...

	// Write bmp file
	//write_bmp(filename04, WIDTH, HEIGHT, (char*)rgb);

	// Some debug
	//printf("GPU color-scale translate + rotate done... --> %s \n", filename04);
	/////////////////////////////////////////////////////////////////////////////////////


	// (4) Translate & Rotate (Advanced version-2) //////////////////////////////////////

	// Get device memory for a 3x3 matrix
	float* dev_m;
	// ...

	// Precalculate the transformation matrix in device memory
	//calculate_traslation_rotation_kernel <<< 1, 1 >>> (dev_m);

	// Get the color for each pixel
	//traslation_rotation_kernel_advanced02 <<< WIDTH, HEIGHT >>> (dev_rgb01, dev_rgb02, dev_m, WIDTH, HEIGHT);

	// Copy the memory to CPU
	// ...

	// Write bmp file
	//write_bmp(filename05, WIDTH, HEIGHT, (char*)rgb);

	// Free transformation matrix memory
	// ...

	// Some debug
	//printf("GPU color-scale translate + rotate done... --> %s \n", filename05);
	/////////////////////////////////////////////////////////////////////////////////////


	// Free GPU memory
	cudaFree(dev_rgb01);

	// Free CPU memory
	cudaFree(dev_rgb02);
}



///////////////////////////////////////////////////////////////////////////////////////////////////
// Main entry point
///////////////////////////////////////////////////////////////////////////////////////////////////


int main()
{
	// Sesion 01
	//sample_01_GPU();
	//sample_01_CPU();
	//sample_02_GPU_mandelbrot();

	// Sesion 02
	//sample_03_GPU_thresholding();
	sample_03_GPU_geom_tx();

	return EXIT_SUCCESS;
}