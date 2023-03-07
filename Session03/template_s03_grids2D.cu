/**
 * EEAV 2022 - Template for Sesion 03
 *
 * @author Miguel Rodrigo-Bort
 * @author Antonio Mart�nez
 * @author Santiago Jim�nez-Serrano
 */

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

#define BMP_HEADER_SIZE 1078


// Constants
#define mmNUM_BLOCKS_X        32
#define mmNUM_BLOCKS_Y        32
#define mmTHREADS_PER_BLOCK_X 32
#define mmTHREADS_PER_BLOCK_Y 32

#define BLOCK_SIZE 32
#define BLOCK_W (mmTHREADS_PER_BLOCK_X + 2)
#define BLOCK_H (mmTHREADS_PER_BLOCK_Y + 2)


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

void read_bmp(const char* path, int* width_out, int* heigth_out, byte** data_out)
{
	FILE* f;

	// Read header
	char m_header[BMP_HEADER_SIZE];
	if (fopen_s(&f, path, "rb") != 0)
	{
		fprintf(stderr, "Unable to open BMP file: %s \n", path);
		return;
	}
	fread(m_header, sizeof(char), BMP_HEADER_SIZE, f);

	// Get image size
	width_out[0]  = *(int*)&m_header[18];
	heigth_out[0] = *(int*)&m_header[22];
	int n = width_out[0] * heigth_out[0];

	// Allocate CPU memory
	*data_out = new byte[n];

	// Read pixels data
	fread(*data_out, sizeof(byte), n, f);
	fclose(f);	

	// Apply some operations in order to get the pixels in the correct order
	byte* vv = *data_out;
	int w    = width_out[0];
	int h    = heigth_out[0];
	int nw = w / 2;

	// Rotate image 180 degrees
	for (int x = 0; x < nw; x++)
	{
		for (int y = 0; y < h; y++)
		{
			int x2 = (w - 1 - x);
			int y2 = (h - 1 - y);

			int ipos1 = x  + y  * w;
			int ipos2 = (x2 + y2 * w);

			byte v1 = vv[ipos1];
			byte v2 = vv[ipos2];

			vv[ipos1] = v2;
			vv[ipos2] = v1;
		}
	}

	// Flip X-axis
	for (int x = 0; x < nw; x++)
	{
		for (int y = 0; y < h; y++)
		{
			int x2 = (w - 1 - x);
			int y2 = y;

			int ipos1 = x + y * w;
			int ipos2 = (x2 + y2 * w);

			byte v1 = vv[ipos1];
			byte v2 = vv[ipos2];

			vv[ipos1] = v2;
			vv[ipos2] = v1;
		}
	}	
}

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
	
	fwrite(&bmph.bfType,          2, 1, file);
	fwrite(&bmph.bfSize,          4, 1, file);
	fwrite(&bmph.bfReserved,      4, 1, file);
	fwrite(&bmph.bfOffBits,       4, 1, file);
	fwrite(&bmph.biSize,          4, 1, file);
	fwrite(&bmph.biWidth,         4, 1, file);
	fwrite(&bmph.biHeight,        4, 1, file);
	fwrite(&bmph.biPlanes,        2, 1, file);
	fwrite(&bmph.biBitCount,      2, 1, file);
	fwrite(&bmph.biCompression,   4, 1, file);
	fwrite(&bmph.biSizeImage,     4, 1, file);
	fwrite(&bmph.biXPelsPerMeter, 4, 1, file);
	fwrite(&bmph.biYPelsPerMeter, 4, 1, file);
	fwrite(&bmph.biClrUsed,       4, 1, file);
	fwrite(&bmph.biClrImportant,  4, 1, file);

	line = (char*)malloc(bytesPerLine);

	for (i = height - 1; i >= 0; i--)
	{
		for (j = 0; j < width; j++)
		{
			ipos = 3 * (width * i + j);
			line[3 * j]     = rgb[ipos + 2];
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
// Ejercicios 2-03 - Gray & Color Scale images (GPU) - (GRID 2D)
///////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void gray_scale_kernel_2D(byte* rgb, int w, int h)
{
	//int x = blockIdx.x * blockDim.x + threadIdx.x;
	//int y = blockIdx.y * blockDim.y + threadIdx.y;

	//int Gray = ...;
	//int ipos = 3 * (w * y + x);

	//rgb[ipos]     = ...;   //R
	//rgb[ipos + 2] = ...;   //B
	//rgb[ipos + 1] = ...;	//G
}

__global__ void color_scale_kernel_2D(byte* rgb, int w, int h)
{
	//int x = ...;
	//int y = ...;
	
	//int ipos = 3 * (w * y + x);

	//rgb[ipos]     = ...; //R
	//rgb[ipos + 1] = ...; //G
	//rgb[ipos + 2] = ...; //B
}

void ejercicio_02_3_01_GPU_basics()
{
	const char* filename01 = "ej_02_3_01_GPU_grayscale.bmp";
	const char* filename02 = "ej_02_3_01_GPU_colorscale.bmp";

	// Allocate CPU memory
	byte* rgb = new byte[IMAGE_SIZE_RGB];

	// GPU pointer
	byte* dev_rgb;

	// Allocate GPU memory
	// ...


	// Define BLOCK_SIZE AND GRID_SIZE
	//int NUM_BLOCKS_X        = ...;
	//int NUM_BLOCKS_Y        = ...;
	//int THREADS_PER_BLOCK_X = ...;
	//int THREADS_PER_BLOCK_Y = ...;
	//dim3 dim_grid(NUM_BLOCKS_X, NUM_BLOCKS_Y);
	//dim3 dim_block(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);


	// Gray scale sample ///////////////////////////////////////////////////////////

	// Get the color for each pixel
	// gray_scale_kernel_2D <<< dim_grid, dim_block >>> (dev_rgb, WIDTH, HEIGHT);

	// Copy the memory to CPU
	// ...
	
	// Write bmp file
	write_bmp(filename01, WIDTH, HEIGHT, (char*)rgb);

	// Some debug
	printf("ejercicio_02_3_01_GPU_basics gray-scale done... --> %s \n", filename01);
	//////////////////////////////////////////////////////////////////////////////


	// Color scale sample ///////////////////////////////////////////////////////////

	// Get the color for each pixel
	//color_scale_kernel_2D <<< dim_grid, dim_block >>> (dev_rgb, WIDTH, HEIGHT);

	// Copy the memory to CPU
	// ...

	// Write bmp file
	//write_bmp(filename02, WIDTH, HEIGHT, (char*)rgb);

	// Some debug
	//printf("ejercicio_02_3_01_GPU_basics color-scale done... --> %s \n", filename02);
	//////////////////////////////////////////////////////////////////////////////

	
	// Free GPU memory
	// ...

	// Free CPU memory
	// ...
}



///////////////////////////////////////////////////////////////////////////////////////////////////
// Ejercicios 2-03 - Mandelbrot Sample - (GRID 2D)
///////////////////////////////////////////////////////////////////////////////////////////////////


__device__ void gray_2_rgb(byte gray, byte* rgb)
{
	// Convert from gray to rgb	
	rgb[0] = gray; //R
	rgb[1] = gray; //G
	rgb[2] = gray; //B
}

__device__ float interpolate(float val, float y0, float x0, float y1, float x1)
{
	// From https://stackoverflow.com/questions/7706339/grayscale-to-red-green-blue-matlab-jet-color-scale
	// ...
	return 0.0f;
}

__device__ float base(float val)
{
	// From https://stackoverflow.com/questions/7706339/grayscale-to-red-green-blue-matlab-jet-color-scale
	// ...

	return 0.0f;
}

__device__ void gray_2_rgb_jet(byte gray, byte* rgb)
{
	// Escale from [0-255] to [0.0-1.0]
	//float g = ((float)gray) / 255.0f;

	// From https://stackoverflow.com/questions/7706339/grayscale-to-red-green-blue-matlab-jet-color-scale
	//float fr = ...;
	//float fg = ...;
	//float fb = ...;

	// Convert from gray to rgb, rescaling to [0-255]
	//rgb[0] = (byte)(fr * 255.0f); //R
	//rgb[1] = (byte)(fg * 255.0f); //G
	//rgb[2] = (byte)(fb * 255.0f); //B
}

__device__ void gray_2_rgb_hot(byte gray, byte* rgb)
{
	// HOT colormap	
	rgb[0] = (gray > 85)   ? 255 :  gray * 3;                          //R
	rgb[1] = (gray > 85*2) ? 255 : (gray < 85) ? 0 : (gray - 85) * 3;  //G
	rgb[2] = (gray < 85*2) ?   0 : (gray - 2 * 85) * 3;                //B
}

__global__ void mandelbrot_kernel_2D(byte* rgb, int W, int H, int iteraciones, int limite, int color_scale)
{
	//int i = ...;
	//int j = ...;	

	//float x = ((float)i) / (W * 1.0f);
	//float y = ((float)j) / (H * 1.0f);
	
	//float z[2] = ...;
	//float w[2] = ...;
	//float c[2] = ...;
	//float az;
	//int   gray;

	// Loop...
	//for (int ij = 0; ij < iteraciones; ij++)
	//{
	// ...
	//}

	// Convert from gray-scale (0-iteraciones) to gray-scale(0-255)
	//byte gray_byte = gray * 255 / iteraciones;

	// Get the pixel position in the array
	//int ipos = 3 * (W * j + i);

	// Save to rgb
	switch (color_scale)
	{
	case 0:
		// Normal Gray-Scale
		//gray_2_rgb(gray_byte, rgb + ipos);
		break;

	case 1:
		// Use the JET color-scale
		//gray_2_rgb_jet(gray_byte, rgb + ipos);
		break;

	default:
		// Use the HOT color-scale
		//gray_2_rgb_hot(gray_byte, rgb + ipos);
		break;
	}
}

void ejercicio_02_3_02_GPU_mandelbrot()
{
	// Output filename	
	const char* filename_prefix = "ejercicio_02_3_02_GPU_mandelbrot.";
	
	// Constants
	const int w = WIDTH;
	const int h = HEIGHT;

	// Allocate CPU memory
	byte* rgb = new byte[IMAGE_SIZE_RGB];

	// GPU pointer
	byte* dev_rgb;

	// Allocate GPU memory
	// ...

	// Define BLOCK_SIZE AND GRID_SIZE
	//int NUM_BLOCKS_X = ...;
	//int NUM_BLOCKS_Y = ...;
	//int THREADS_PER_BLOCK_X = ...;
	//int THREADS_PER_BLOCK_Y = ...;
	//dim3 dim_grid(NUM_BLOCKS_X, NUM_BLOCKS_Y);
	//dim3 dim_block(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);


	// Mandelbrot /////////////////////////////////////////////////////////////////

	// Mandelbrot params
	int iteraciones = 20;
	int limite      =  2;

	// For each color scale...
	for (int cs = 0; cs < 3; cs++)
	{
		// Get the color for each pixel	
		// mandelbrot_kernel_2D <<< dim_grid, dim_block >>> (dev_rgb, w, h, iteraciones, limite, cs);

		// Copy the memory to CPU
		// ...

		// Write bmp file
		std::string s(filename_prefix);
		s.append(std::to_string(cs)).append(".bmp");;
		write_bmp(s.c_str(), w, h, (char*)rgb);

		// Some debug
		printf("ejercicio_02_3_02_GPU_mandelbrot done... --> %s \n", s.c_str());
	}

	//////////////////////////////////////////////////////////////////////////////

	// Free GPU memory
	// ...

	// Free CPU memory
	// ...
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// Ejercicios 3-01 - Erode + Threshold (GRID 2D)
///////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void threshold_kernel_2d(byte* img_gray_in, byte* img_gray_out, int w, int h, byte th1, byte th2)
{
	// Get the x and y coordinates from the image using threadIdx and blockIdx values
	int x = threadIdx.x + (blockIdx.x * blockDim.x);
	int y = threadIdx.y + (blockIdx.y * blockDim.y);

	// Check image limits
	if (x >= w || y >= h)
		return;

	// Get the pixel position in the array
	int ipos = ((w * y) + x);

	// Get the current gray value
	byte value = img_gray_in[ipos];

	if (value > th1 && value < th2)
	{
		// Set the true value (255)
		img_gray_out[ipos] = 255;

	}
	else
	{
		// Set the false value (0)
		img_gray_out[ipos] = 0;
	}
}

__global__ void erode_3x3_kernel_2D(byte* img_gray_in, byte* img_gray_out, int w, int h) 
{
	// Get the x and y coordinates from the image using threadIdx and blockIdx values
	int x = threadIdx.x + (blockIdx.x * blockDim.x);
	int y = threadIdx.y + (blockIdx.y * blockDim.y);

	// Check image limits
	if (x >= w || y >= h)
		return;

	// Get the array index with x & y coordinates
	int ipos = ((y*w) + x);

	// Get the actual value in the input image
	byte value = img_gray_in[ipos];

	// Warning with the borders!!!
	for (int xx = x-1; xx <= x+1; xx++)
	{
		for (int yy = y-1; yy <= y+1; yy++)
		{
			if (xx < 0 || xx >= w || yy < 0 || yy >= h)
				return;
			if (img_gray_in[(yy * w) + xx] == 0)
				value = 0;
		}
	}

	// Set the output value
	img_gray_out[ipos] = value;
}

__device__ void set_gray_value(byte* rgb, byte gray_value)
{
	rgb[0] = gray_value;
	rgb[1] = gray_value;
	rgb[2] = gray_value;
}

__global__ void gray_to_rgb_kernel_2D(byte* img_gray_in, byte* img_rgb_out, int w, int h) 
{
	// Get the x and y coordinates from the image using threadIdx and blockIdx values
	int x = threadIdx.x + (blockIdx.x * blockDim.x);
	int y = threadIdx.y + (blockIdx.y * blockDim.y);

	// Check image limits
	if (x >= w || y >= h)
		return;

	// Get the array index with x & y coordinates
	int ipos = x + (y * w);

	// Get the gray value from the input image
	byte grayLevel = img_gray_in[ipos];

	// Set the gray value to the output image
	set_gray_value(img_rgb_out + (ipos * 3), grayLevel);
}

void ejercicio_03_1_01_GPU_th_erode()
{
	// Output filename	
	const char* filename01 = "ejercicio_03_1_01_GPU_th.bmp";
	const char* filename02 = "ejercicio_03_1_01_GPU_th_erode.bmp";


	// Read input bmp file
	const char* filename_in = "lena_gray.bmp";
	int w = 0;
	int h = 0;
	byte* grayscale_in = NULL;
	read_bmp(filename_in, &w, &h, &grayscale_in);
	int gray_img_sz = w * h;
	int rgb__img_sz = w * h * 3;
	printf("Read done!\nW: %d, H: %d \n", w, h);


	// Allocate CPU memory
	byte* rgb_out02 = new byte[rgb__img_sz];
	byte* rgb_out03 = new byte[rgb__img_sz];

	// Define BLOCK_SIZE AND GRID_SIZE
	int NUM_BLOCKS_X = mmNUM_BLOCKS_X;
	int NUM_BLOCKS_Y = mmNUM_BLOCKS_Y;
	int THREADS_PER_BLOCK_X = mmTHREADS_PER_BLOCK_X;
	int THREADS_PER_BLOCK_Y = mmTHREADS_PER_BLOCK_Y;
	dim3 dim_grid(NUM_BLOCKS_X, NUM_BLOCKS_Y);
	dim3 dim_block(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);


	// GPU pointers
	byte* dev_gray_01;
	byte* dev_gray_02;
	byte* dev_gray_03;
	byte* dev_rgb02;
	byte* dev_rgb03;


	// Allocate GPU memory
	cudaMalloc((void**)&dev_gray_01, gray_img_sz * sizeof(byte));
	cudaMalloc((void**)&dev_gray_02, gray_img_sz * sizeof(byte));
	cudaMalloc((void**)&dev_gray_03, gray_img_sz * sizeof(byte));
	cudaMalloc((void**)&dev_rgb02, rgb__img_sz * sizeof(byte));
	cudaMalloc((void**)&dev_rgb03, rgb__img_sz * sizeof(byte));


	// Copy input data to GPU memory
	cudaMemcpy(dev_gray_01, grayscale_in, gray_img_sz * sizeof(byte), cudaMemcpyHostToDevice);


	// Thresholding values
	byte th1 = 100;
	byte th2 = 127;


	// Kernels calling with gray scale images
	threshold_kernel_2d << < dim_grid, dim_block >> > (dev_gray_01, dev_gray_02, w, h, th1, th2);
	erode_3x3_kernel_2D <<< dim_grid, dim_block >>> (dev_gray_02, dev_gray_03, w, h);

	// Kernels calling to convert gray to rgb images
	gray_to_rgb_kernel_2D <<< dim_grid, dim_block >>> (dev_gray_02, dev_rgb02, w, h);
	gray_to_rgb_kernel_2D <<< dim_grid, dim_block >>> (dev_gray_03, dev_rgb03, w, h);


	// Copy the results to to CPU memory
	cudaMemcpy(rgb_out02, dev_rgb02, rgb__img_sz * sizeof(byte), cudaMemcpyDeviceToHost);
	cudaMemcpy(rgb_out03, dev_rgb03, rgb__img_sz * sizeof(byte), cudaMemcpyDeviceToHost);



	// Write bmp file
	write_bmp(filename01, w, h, (char*)rgb_out02);
	write_bmp(filename02, w, h, (char*)rgb_out03);


	// Some debug
	printf("ejercicio_03_1_01_GPU_th_erode done... --> %s \n", filename01);


	// Free GPU memory
	cudaFree(dev_gray_01);
	cudaFree(dev_gray_02);
	cudaFree(dev_gray_03);
	cudaFree(dev_rgb02);
	cudaFree(dev_rgb03);
	// Free CPU memory
	delete[] rgb_out02;
	delete[] rgb_out03;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Ejercicios 3-02 - Filter 3X3
///////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void filter_3x3_kernel_2D(byte* img_gray_in, byte* img_gray_out, int* filter, int w, int h) 
{
	// Get the x and y coordinates from the image using threadIdx and blockIdx values
	int x = threadIdx.x + (blockIdx.x * blockDim.x);
	int y = threadIdx.y + (blockIdx.y * blockDim.y);

	// Check image limits
	if (x >= w || y >= h)
		return;

	// Get the array index with x & y coordinates in the image
	int ipos = ((y*w) + x);

	// Pixel value
	int pixel = 0;

	// Apply the filter - Warning with the borders!!!
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			int xx = x + j - 1;
			int yy = y + i - 1;
			if (xx >= w || yy >= h) return;
			int iipos = xx + (yy * w);
			pixel += filter[(3 * i) + j] * img_gray_in[iipos];
		}
	}

	// Normalize the pixel value to (0-255)
	pixel = (pixel / 2) + 128;

	// Save the value to the output image
	img_gray_out[ipos] = pixel;
}


void ejercicio_03_2_02_GPU_filter()
{
	// Output filename		
	const char* filename01 = "ejercicio_03_2_02_GPU_filter.bmp";


	// Read input bmp file
	const char* filename_in = "lena_gray.bmp";
	int w = 0;
	int h = 0;
	byte* grayscale_in = NULL;
	read_bmp(filename_in, &w, &h, &grayscale_in);
	int gray_img_sz = w * h;
	int rgb__img_sz = w * h * 3;
	printf("Read done!\nW: %d, H: %d \n", w, h);


	// Allocate CPU memory
	byte* rgb_out02 = new byte[rgb__img_sz];
	int* filter     = new int[9]{ 0, 0, 0, 0, 1, -1, 0, 0, 0 };

	// Define BLOCK_SIZE AND GRID_SIZE
	int NUM_BLOCKS_X = mmNUM_BLOCKS_X;
	int NUM_BLOCKS_Y = mmNUM_BLOCKS_Y;
	int THREADS_PER_BLOCK_X = mmTHREADS_PER_BLOCK_X;
	int THREADS_PER_BLOCK_Y = mmTHREADS_PER_BLOCK_Y;
	dim3 dim_grid(NUM_BLOCKS_X, NUM_BLOCKS_Y);
	dim3 dim_block(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);


	// GPU pointers
	byte* dev_gray_01;
	byte* dev_gray_02;	
	byte* dev_rgb02;
	int*  dev_filter;


	// Allocate GPU memory
	cudaMalloc((void**)&dev_gray_01, gray_img_sz * sizeof(byte));
	cudaMalloc((void**)&dev_gray_02, gray_img_sz * sizeof(byte));
	cudaMalloc((void**)&dev_filter, 9 * sizeof(int));
	cudaMalloc((void**)&dev_rgb02, rgb__img_sz * sizeof(byte));


	// Copy input data to GPU memory
	cudaMemcpy(dev_gray_01, grayscale_in, gray_img_sz * sizeof(byte), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_filter, filter, 9 * sizeof(int), cudaMemcpyHostToDevice);


	// Kernels calling with gray scale images	
	filter_3x3_kernel_2D <<< dim_grid, dim_block >>> (dev_gray_01, dev_gray_02, dev_filter, w, h);


	// Kernels calling to convert gray to rgb images
	gray_to_rgb_kernel_2D <<< dim_grid, dim_block >>> (dev_gray_02, dev_rgb02, w, h);


	// Copy the results to to CPU memory
	cudaMemcpy(rgb_out02, dev_rgb02, rgb__img_sz * sizeof(byte), cudaMemcpyDeviceToHost);


	// Write bmp file
	write_bmp(filename01, w, h, (char*)rgb_out02);


	// Some debug
	printf("ejercicio_03_2_02_GPU_filter done... --> %s \n", filename01);


	// Free GPU memory
	cudaFree(dev_gray_01);
	cudaFree(dev_gray_02);
	cudaFree(dev_rgb02);
	cudaFree(dev_filter);
	// Free CPU memory
	delete[] rgb_out02;
}





///////////////////////////////////////////////////////////////////////////////////////////////////
// Ejercicios 3-02 - Erode 3X3 WITH SHARED MEMORY
///////////////////////////////////////////////////////////////////////////////////////////////////

__device__ int get_img_idx(int x, int y, int w)
{
	return (y * w) + x;
}

__device__ void copy_border_block(byte* img_gray_in, byte* img_gray_sh, int w, int h, int x, int y, int x_sh, int y_sh)
{
	// Avoid do nothing inside the borders
	if (threadIdx.x != 0 && threadIdx.x != blockDim.x - 1 &&
		threadIdx.y != 0 && threadIdx.y != blockDim.y - 1)
	{
		return;
	}

	// Local variables
	int esquina_superior_izquierda = 0;
	int esquina_superior_derecha = BLOCK_W - 1;
	int esquina_inferior_izquierda = BLOCK_W * (BLOCK_H - 1); //(BLOCK_W * BLOCK_H) - BLOCK_W
	int esquina_inferior_derecha = (BLOCK_H * BLOCK_W) - 1;

	////////////////////////////////////////
	// Copy x-axis borders
	if (threadIdx.x == 0)
	{
		img_gray_sh[(x_sh - 1) + (y_sh * BLOCK_W)] = img_gray_in[(x - 1) + (y * w)];
	}
	else if (threadIdx.x == blockDim.x - 1)
	{
		img_gray_sh[(x_sh + 1) + (y_sh * BLOCK_W)] = img_gray_in[(x + 1) + (y * w)];
	}

	
	////////////////////////////////////////
	// Copy y-axis borders
	if (threadIdx.y == 0)
	{
		img_gray_sh[x_sh + ((y_sh - 1) * BLOCK_W)] = img_gray_in[x + ((y - 1) * w)];
	}
	else if (threadIdx.y == blockDim.y - 1)
	{
		img_gray_sh[x_sh + ((y_sh + 1) * BLOCK_W)] = img_gray_in[x + ((y + 1) * w)];
	}
	

	/////////////////////////////////////
	// Copy 4 corners pixels
	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		img_gray_sh[esquina_superior_izquierda] = img_gray_in[(x - 1) + ((y - 1) * w)];
	}
	else if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1)
	{
		img_gray_sh[esquina_inferior_izquierda] = img_gray_in[(x - 1) + ((y + 1) * w)];
	}
	else if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0)
	{
		img_gray_sh[esquina_superior_derecha] = img_gray_in[(x + 1) + ((y - 1) * w)];
	}
	else if (threadIdx.x == blockDim.x - 1  && threadIdx.y == blockDim.y - 1)
	{
		img_gray_sh[esquina_inferior_derecha] = img_gray_in[(x + 1) + ((y + 1) * w)];
	}
}

__global__ void erode_3x3_kernel_2D_shared_mem(byte* img_gray_in, byte* img_gray_out, int w, int h) 
{
	// Get the x and y coordinates from the image using threadIdx and blockIdx values
	int x = threadIdx.x + (blockIdx.x * blockDim.x);
	int y = threadIdx.y + (blockIdx.y * blockDim.y);

	// Get the coordinates in the shared memory block
	int x_sh = threadIdx.x + 1;
	int y_sh = threadIdx.y + 1;

	// Check image limits
	if (x >= w || y >= h)
		return;

	// Get the array index with x & y coordinates in the image
	int ipos = (y * w) + x;

	// Get the array index within the shared memory block
	int ipos_sh = (y_sh * BLOCK_W) + x_sh;

	// Pixel value
	byte pixel = 0;

	// Declare the shared memory needed
	__shared__ byte img_gray_sh[BLOCK_W * BLOCK_H];

	// Set the values from the input image to the shared image (no borders taked into account)
	img_gray_sh[ipos_sh] = img_gray_in[ipos];

	// Copy the memory outside the central part of the block --> borders of the mark of the 
	copy_border_block(img_gray_in, img_gray_sh, w, h, x, y, x_sh, y_sh);

	// Wait for all the threads to finish image copy to shared memory
	__syncthreads();

	// Apply the erode
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++) 
		{
			int xx_sh = x_sh + j - 1;
			int yy_sh = y_sh + i - 1;
			if (img_gray_sh[(yy_sh * BLOCK_W) + xx_sh] == pixel)
			{
				img_gray_sh[ipos_sh] = pixel;
				return;
			}
		}
	}

	// Save the value to the output image
	img_gray_out[ipos] = img_gray_sh[ipos_sh];
}


void ejercicio_03_2_01_GPU_th_erode_shared_mem()
{
	// Output filename	
	const char* filename01 = "ejercicio_03_2_01_GPU_th.bmp";
	const char* filename02 = "ejercicio_03_2_01_GPU_th_erode_shared_mem.bmp";


	// Read input bmp file
	const char* filename_in = "lena_gray.bmp";
	int w = 0;
	int h = 0;
	byte* grayscale_in = NULL;
	read_bmp(filename_in, &w, &h, &grayscale_in);
	int gray_img_sz = w * h;
	int rgb__img_sz = w * h * 3;
	printf("Read done!\nW: %d, H: %d \n", w, h);


	// Allocate CPU memory
	byte* rgb_out02 = new byte[rgb__img_sz];
	byte* rgb_out03 = new byte[rgb__img_sz];

	// Define BLOCK_SIZE AND GRID_SIZE
	dim3 dim_grid(mmNUM_BLOCKS_X, mmNUM_BLOCKS_Y);
	dim3 dim_block(mmTHREADS_PER_BLOCK_X, mmTHREADS_PER_BLOCK_Y);


	// GPU pointers
	byte* dev_gray_01;
	byte* dev_gray_02;
	byte* dev_gray_03;
	byte* dev_rgb02;
	byte* dev_rgb03;


	// Allocate GPU memory
	cudaMalloc((void**)&dev_gray_01, gray_img_sz * sizeof(byte));
	cudaMalloc((void**)&dev_gray_02, gray_img_sz * sizeof(byte));
	cudaMalloc((void**)&dev_gray_03, gray_img_sz * sizeof(byte));
	cudaMalloc((void**)&dev_rgb02, rgb__img_sz * sizeof(byte));
	cudaMalloc((void**)&dev_rgb03, rgb__img_sz * sizeof(byte));


	// Copy input data to GPU memory
	cudaMemcpy(dev_gray_01, grayscale_in, gray_img_sz * sizeof(byte), cudaMemcpyHostToDevice);


	// Thresholding values
	byte th1 = 100;
	byte th2 = 127;

	// Kernels calling with gray scale images
	threshold_kernel_2d <<< dim_grid, dim_block >>> (dev_gray_01, dev_gray_02, w, h, th1, th2);
	erode_3x3_kernel_2D_shared_mem <<< dim_grid, dim_block >>> (dev_gray_02, dev_gray_03, w, h);

	// Kernels calling to convert gray to rgb images
	gray_to_rgb_kernel_2D <<< dim_grid, dim_block >>> (dev_gray_02, dev_rgb02, w, h);
	gray_to_rgb_kernel_2D <<< dim_grid, dim_block >>> (dev_gray_03, dev_rgb03, w, h);


	// Copy the results to to CPU memory
	cudaMemcpy(rgb_out02, dev_rgb02, rgb__img_sz * sizeof(byte), cudaMemcpyDeviceToHost);
	cudaMemcpy(rgb_out03, dev_rgb03, rgb__img_sz * sizeof(byte), cudaMemcpyDeviceToHost);


	// Write bmp file
	write_bmp(filename01, w, h, (char*)rgb_out02);
	write_bmp(filename02, w, h, (char*)rgb_out03);


	// Some debug
	printf("ejercicio_03_1_01_GPU_th_erode done... --> %s \n", filename01);


	// Free GPU memory
	cudaFree(dev_gray_01);
	cudaFree(dev_gray_02);
	cudaFree(dev_gray_03);
	cudaFree(dev_rgb02);
	cudaFree(dev_rgb03);
	// Free CPU memory
	delete[] rgb_out02;
	delete[] rgb_out03;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// Ejercicios 3-02 - Filter 3X3 WITH SHARED MEMORY
///////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void filter_3x3_kernel_2D_shared_mem(byte* img_gray_in, byte* img_gray_out, int* filter, int w, int h) 
{
	// Get the x and y coordinates from the image using threadIdx and blockIdx values
	int x = threadIdx.x + (blockIdx.x * blockDim.x);
	int y = threadIdx.y + (blockIdx.y * blockDim.y);

	// Get the coordinates in the shared memory block
	int x_sh = threadIdx.x + 1;
	int y_sh = threadIdx.y + 1;

	// Check image limits
	if (x >= w || y >= h)
		return;

	// Get the array index with x & y coordinates in the image
	int ipos = (y * w) + x;

	// Get the array index within the shared memory block
	int ipos_sh = (y_sh * BLOCK_W) + x_sh;

	// Declare the shared memory needed
	__shared__ byte img_gray_sh[BLOCK_W * BLOCK_H];

	// Set the values from the input image to the shared image (no borders taked into account)
	img_gray_sh[ipos_sh] = img_gray_in[ipos];

	// Copy the memory outside the central part of the block --> borders of the mark of the block
	copy_border_block(img_gray_in, img_gray_sh, w, h, x, y, x_sh, y_sh);

	// Wait for all the threads to finish image copy to shared memory
	__syncthreads();

	// Pixel value
	int pixel = 0;

	// Apply the filter
	for (int i = 0; i < 3; i++) 
	{
		for (int j = 0; j < 3; j++) 
		{
			int xx_sh = x_sh + j - 1;
			int yy_sh = y_sh + i - 1;
			//if (xx_sh >= blockDim.x || yy >= blockDim.y) return;
			int iipos_sh = xx_sh + (yy_sh * BLOCK_W);
			pixel += filter[(3 * i) + j] * img_gray_sh[iipos_sh];
		}
	}

	// Normalize the pixel value to (0-255)
	pixel = (pixel / 2) + 128;

	// Save the value to the output image
	img_gray_out[ipos] = pixel;
}


void ejercicio_03_2_02_GPU_filter_shared_mem()
{
	// Output filename		
	const char* filename01 = "ejercicio_03_2_02_GPU_filter_shared_mem.bmp";


	// Read input bmp file
	const char* filename_in = "lena_gray.bmp";
	int w = 0;
	int h = 0;
	byte* grayscale_in = NULL;
	read_bmp(filename_in, &w, &h, &grayscale_in);
	int gray_img_sz = w * h;
	int rgb__img_sz = w * h * 3;
	printf("Read done!\nW: %d, H: %d \n", w, h);


	// Allocate CPU memory
	byte* rgb_out02 = new byte[rgb__img_sz];
	int* filter     = new int[9]{ 0, 0, 0, 0, 1, -1, 0, 0, 0 };

	// Define BLOCK_SIZE AND GRID_SIZE
	dim3 dim_grid(mmNUM_BLOCKS_X,         mmNUM_BLOCKS_Y);
	dim3 dim_block(mmTHREADS_PER_BLOCK_X, mmTHREADS_PER_BLOCK_Y);


	// GPU pointers
	byte* dev_gray_01;
	byte* dev_gray_02;	
	byte* dev_rgb02;
	int*  dev_filter;


	// Allocate GPU memory
	cudaMalloc((void**)&dev_gray_01, gray_img_sz * sizeof(byte));
	cudaMalloc((void**)&dev_gray_02, gray_img_sz * sizeof(byte));
	cudaMalloc((void**)&dev_filter, 9 * sizeof(int));
	cudaMalloc((void**)&dev_rgb02, rgb__img_sz * sizeof(byte));


	// Copy input data to GPU memory
	cudaMemcpy(dev_gray_01, grayscale_in, gray_img_sz * sizeof(byte), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_filter, filter, 9 * sizeof(int), cudaMemcpyHostToDevice);


	// Kernels calling with gray scale images	
	filter_3x3_kernel_2D_shared_mem <<< dim_grid, dim_block >>> (dev_gray_01, dev_gray_02, dev_filter, w, h);


	// Kernels calling to convert gray to rgb images
	gray_to_rgb_kernel_2D <<< dim_grid, dim_block >>> (dev_gray_02, dev_rgb02, w, h);


	// Copy the results to to CPU memory
	cudaMemcpy(rgb_out02, dev_rgb02, rgb__img_sz * sizeof(byte), cudaMemcpyDeviceToHost);


	// Write bmp file
	write_bmp(filename01, w, h, (char*)rgb_out02);


	// Some debug
	printf("ejercicio_03_2_02_GPU_filter_shared_mem done... --> %s \n", filename01);


	// Free GPU memory
	cudaFree(dev_gray_01);
	cudaFree(dev_gray_02);
	cudaFree(dev_rgb02);
	cudaFree(dev_filter);

	// Free CPU memory
	delete[] rgb_out02;
}




///////////////////////////////////////////////////////////////////////////////////////////////////
// Main entry point
///////////////////////////////////////////////////////////////////////////////////////////////////


int main()
{
	// Sesion 02
	//ejercicio_02_3_01_GPU_basics();
	//ejercicio_02_3_02_GPU_mandelbrot();

	// Sesion 03
	ejercicio_03_1_01_GPU_th_erode();
	ejercicio_03_2_02_GPU_filter();

	// Sesion 03 with shared mem
	ejercicio_03_2_01_GPU_th_erode_shared_mem();
	ejercicio_03_2_02_GPU_filter_shared_mem();

	return EXIT_SUCCESS;
}
