#include "Utils.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>


#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/extrema.h>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////// Fusion two images and detect intersection //////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

cudaError_t cuInit()
{
	cudaError_t cudaStatus = cudaSuccess;
	cudaSetDevice(0);
	return cudaStatus;
}

cudaError_t cuDestory()
{
	cudaError_t cudaStatus = cudaSuccess;
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
		printf("cudaDeviceReset failed!");

	return cudaStatus;
}


//background color should be 0,0,0
struct min_compare_with_exclude
{
	__host__ __device__
		bool operator()(int lhs, int rhs)
	{
		if (lhs != -1e9 && rhs != -1e9)				// -1e9 is a background color marker.
			return lhs < rhs;
		else if (lhs == -1e9 && rhs != -1e9)
			return false;
		else if (lhs != -1e9 && rhs == -1e9)
			return true;
		else
			return lhs < rhs;
	}
};

__global__ void cuMarkColorPixel(uint8* image, int rows, int cols, int* color_mark_array)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total_thread = blockDim.x*gridDim.x;

	for (int i = idx; i<rows*cols; i += total_thread)
	{
		int current_row = i / cols;
		int current_col = i%cols;
		// pixel position
		int pixel_r_index = current_row*cols * 3 + current_col * 3;
		int pixel_g_index = current_row*cols * 3 + current_col * 3 + 1;
		int pixel_b_index = current_row*cols * 3 + current_col * 3 + 2;

		// r,g,b
		uint8 r = image[pixel_r_index];
		uint8 g = image[pixel_g_index];
		uint8 b = image[pixel_b_index];
		if (r == 0 && g == 0 && b == 0)
			color_mark_array[i] = 0;
		else
			color_mark_array[i] = 1;
	}
}

__global__ void cuFindColorPixelIndice(int* row_index_array, int* col_index_array, uint8* image, int rows, int cols)
{
	// blockdim = rows, 
	//unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total_thread = blockDim.x*gridDim.x;

	for (int i = idx; i<rows*cols; i += total_thread)
	{
		int current_row = i / cols;
		int current_col = i%cols;
		// pixel position
		int pixel_r_index = current_row*cols * 3 + current_col * 3;
		int pixel_g_index = current_row*cols * 3 + current_col * 3 + 1;
		int pixel_b_index = current_row*cols * 3 + current_col * 3 + 2;

		// r,g,b
		uint8 r = image[pixel_r_index];
		uint8 g = image[pixel_g_index];
		uint8 b = image[pixel_b_index];


		if (r == 0 && g == 0 && b == 0)
		{
			row_index_array[current_row*cols + current_col] = -1e9;
			col_index_array[current_row*cols + current_col] = -1e9;
		}
		else
		{
			row_index_array[current_row*cols + current_col] = current_row;
			col_index_array[current_row*cols + current_col] = current_col;
		}
	}
}

__global__ void cuTransformPixelIndice(int* row_index_array, int* col_index_array, int array_length, int* new_row_index_array, int* new_col_index_array, double* affine_mat)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total_thread = blockDim.x*gridDim.x;
	for (int i = idx; i < array_length; i += total_thread)
	{
		int row = row_index_array[i];
		int col = col_index_array[i];
		if (row != -1e9)
		{
			int new_row = (int)(affine_mat[0] * (double)row + affine_mat[1] * (double)col + affine_mat[2]);
			int new_col = (int)(affine_mat[3] * (double)row + affine_mat[4] * (double)col + affine_mat[5]);
			new_row_index_array[i] = new_row;
			new_col_index_array[i] = new_col;
		}
		else
		{
			new_row_index_array[i] = -1e9;
			new_col_index_array[i] = -1e9;
		}
	}
}

cudaError_t LaunchCudaFindBoundingBox(const uint8* image, int rows, int cols, std::vector<UV_i>& out_boundingbox, double* affine/*=NULL*/)
{
	cudaError_t cudaStatus = cudaSuccess;

	uint8* dst_cu;
	int* row_index_array_cu;
	int* col_index_array_cu;

	cudaStatus = cudaMalloc((void**)&dst_cu, rows*cols * 3 * sizeof(uint8));
	cudaStatus = cudaMalloc((void**)&row_index_array_cu, rows*cols * sizeof(int));
	cudaStatus = cudaMalloc((void**)&col_index_array_cu, rows*cols * sizeof(int));

	cudaStatus = cudaMemcpy(dst_cu, image, rows*cols * 3 * sizeof(uint8), cudaMemcpyHostToDevice);
	// find color pixels
	cuFindColorPixelIndice << <1024, 512 >> >(row_index_array_cu, col_index_array_cu, dst_cu, rows, cols);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("cuFindColorPixelIndice launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("cudaDeviceSynchronize returned error code %d after launching cuFindColorPixelIndice!\n", cudaStatus);
		return cudaStatus;
	}
	if (affine == NULL)
	{
		thrust::device_ptr<int> row_array_ptr(row_index_array_cu);
		thrust::device_ptr<int> col_array_ptr(col_index_array_cu);
		// reduction
		int min_row = *(thrust::min_element(row_array_ptr, row_array_ptr + cols*rows, min_compare_with_exclude()));
		int min_col = *(thrust::min_element(col_array_ptr, col_array_ptr + cols*rows, min_compare_with_exclude()));
		int max_row = *(thrust::max_element(row_array_ptr, row_array_ptr + cols*rows));
		int max_col = *(thrust::max_element(col_array_ptr, col_array_ptr + cols*rows));

		UV_i min_px, max_px;
		min_px.row_i = min_row;
		min_px.col_i = min_col;
		max_px.row_i = max_row + 1;
		max_px.col_i = max_col + 1;
		out_boundingbox.push_back(min_px);
		out_boundingbox.push_back(max_px);
	}
	else
	{
		double* affine_mat_cu;
		int* new_row_index_array_cu;
		int* new_col_index_array_cu;
		cudaStatus = cudaMalloc((void**)&affine_mat_cu, 6 * sizeof(double));
		cudaStatus = cudaMalloc((void**)&new_row_index_array_cu, rows*cols * sizeof(int));
		cudaStatus = cudaMalloc((void**)&new_col_index_array_cu, rows*cols * sizeof(int));
		cudaStatus = cudaMemcpy(affine_mat_cu, affine, 6 * sizeof(double), cudaMemcpyHostToDevice);
		cuTransformPixelIndice << <1024, 512 >> >(row_index_array_cu, col_index_array_cu, rows*cols, new_row_index_array_cu, new_col_index_array_cu, affine_mat_cu);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			printf("cuTransformPixelIndice launch failed: %s\n", cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			printf("cudaDeviceSynchronize returned error code %d after launching cuTransformPixelIndice!\n", cudaStatus);
			return cudaStatus;
		}

		thrust::device_ptr<int> new_row_array_ptr(new_row_index_array_cu);
		thrust::device_ptr<int> new_col_array_ptr(new_col_index_array_cu);
		// reduction sort
		int min_row = *(thrust::min_element(new_row_array_ptr, new_row_array_ptr + cols*rows, min_compare_with_exclude()));
		int min_col = *(thrust::min_element(new_col_array_ptr, new_col_array_ptr + cols*rows, min_compare_with_exclude()));
		int max_row = *(thrust::max_element(new_row_array_ptr, new_row_array_ptr + cols*rows));
		int max_col = *(thrust::max_element(new_col_array_ptr, new_col_array_ptr + cols*rows));

		UV_i new_min_px, new_max_px;
		new_min_px.row_i = min_row;
		new_min_px.col_i = min_col;
		new_max_px.row_i = max_row + 1;
		new_max_px.col_i = max_col + 1;
		out_boundingbox.push_back(new_min_px);
		out_boundingbox.push_back(new_max_px);

		cudaFree(affine_mat_cu);
		cudaFree(new_row_index_array_cu);
		cudaFree(new_col_index_array_cu);
	}
	cudaFree(dst_cu);
	cudaFree(row_index_array_cu);
	cudaFree(col_index_array_cu);

	return cudaStatus;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void cuFillNewImage(uint8* dst_brefore_trans_cu, int rows, int cols, UV_i* new_dst_image, int start_row, int start_col, int end_row, int end_col, double* remap_mat)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total_thread = blockDim.x*gridDim.x;

	int new_rows = end_row - start_row;
	int new_cols = end_col - start_col;

	for (int i = idx; i < new_rows*new_cols; i += total_thread)
	{
		int current_row = i / new_cols;
		int current_col = i%new_cols;
		int offset_row = current_row + start_row;
		int offset_col = current_col + start_col;

		// pixel position
		int index = current_row*new_cols + current_col;

		// remap to original position
		int original_row = int(remap_mat[0] * (double)offset_row + remap_mat[1] * (double)offset_col + remap_mat[2]);
		int original_col = int(remap_mat[3] * (double)offset_row + remap_mat[4] * (double)offset_col + remap_mat[5]);
		if (original_row<0 || original_row >= rows || original_col<0 || original_col >= cols)
		{
			// r,g,b
			new_dst_image[index].row_i = offset_row;
			new_dst_image[index].col_i = offset_col;
			new_dst_image[index].b = 0;
			new_dst_image[index].g = 0;
			new_dst_image[index].r = 0;
		}
		else
		{
			int original_r_index = original_row*cols * 3 + original_col * 3;
			int original_g_index = original_row*cols * 3 + original_col * 3 + 1;
			int original_b_index = original_row*cols * 3 + original_col * 3 + 2;

			new_dst_image[index].row_i = offset_row;
			new_dst_image[index].col_i = offset_col;
			new_dst_image[index].r = dst_brefore_trans_cu[original_r_index];
			new_dst_image[index].g = dst_brefore_trans_cu[original_g_index];
			new_dst_image[index].b = dst_brefore_trans_cu[original_b_index];
		}
	}
}

cudaError_t LaunchCudaRemapFillNewImage(const uint8* before_trans_image, int rows, int cols, UV_i new_dst_start_pt, UV_i new_dst_end_pt, double* remap_transform, UV_i*& out_new_dst_image_device_ptr)	// remap_transform has been inversed	
{
	cudaError_t cudaStatus = cudaSuccess;

	int new_rows = new_dst_end_pt.row_i - new_dst_start_pt.row_i;
	int new_cols = new_dst_end_pt.col_i - new_dst_start_pt.col_i;
	int start_row = new_dst_start_pt.row_i;
	int start_col = new_dst_start_pt.col_i;
	int end_row = new_dst_end_pt.row_i;
	int end_col = new_dst_end_pt.col_i;


	uint8* dst_brefore_trans_cu;
	UV_i* new_dst_image;
	double* remap_mat;
	cudaStatus = cudaMalloc((void**)&dst_brefore_trans_cu, rows*cols * 3 * sizeof(uint8));
	cudaStatus = cudaMalloc((void**)&new_dst_image, new_rows*new_cols * sizeof(UV_i));
	cudaStatus = cudaMalloc((void**)&remap_mat, 6 * sizeof(double));
	cudaStatus = cudaMemcpy(dst_brefore_trans_cu, before_trans_image, rows*cols * 3 * sizeof(uint8), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(remap_mat, remap_transform, 6 * sizeof(double), cudaMemcpyHostToDevice);

	// fill new image
	cuFillNewImage << <1024, 512 >> >(dst_brefore_trans_cu, rows, cols, new_dst_image, start_row, start_col, end_row, end_col, remap_mat);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("cuFillNewImage launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("cudaDeviceSynchronize returned error code %d after launching cuFillNewImage!\n", cudaStatus);
		return cudaStatus;
	}

	cudaFree(dst_brefore_trans_cu);
	//cudaFree(new_dst_image);
	out_new_dst_image_device_ptr = new_dst_image;
	cudaFree(remap_mat);

	return cudaStatus;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void cuCalculateOverlap(uint8* src_img_cu, int src_rows, int src_cols, UV_i* new_dst_image_device_ptr, int new_rows, int new_cols, int* out_overlap_pixels)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total_thread = blockDim.x*gridDim.x;

	for (int i = idx; i < new_rows*new_cols; i += total_thread)
	{
		if (!(new_dst_image_device_ptr[i].r == 0 && new_dst_image_device_ptr[i].g == 0 && new_dst_image_device_ptr[i].b == 0))
		{
			int new_dst_row = new_dst_image_device_ptr[i].row_i;
			int new_dst_col = new_dst_image_device_ptr[i].col_i;
			if (new_dst_row >= src_rows || new_dst_row < 0 || new_dst_col >= src_cols || new_dst_col < 0)
				out_overlap_pixels[i] = 0;
			else
			{
				int src_index_r = new_dst_row*src_cols * 3 + new_dst_col * 3;
				int src_index_g = new_dst_row*src_cols * 3 + new_dst_col * 3 + 1;
				int src_index_b = new_dst_row*src_cols * 3 + new_dst_col * 3 + 2;

				if (src_img_cu[src_index_r] == 0 && src_img_cu[src_index_g] == 0 && src_img_cu[src_index_b] == 0)
					out_overlap_pixels[i] = 0;
				else
					out_overlap_pixels[i] = 1;
			}
		}
		else
			out_overlap_pixels[i] = 0;
	}
}
__global__ void cuMarkColorPixelUVI(UV_i* new_dst_image_device_ptr, int new_rows, int new_cols, int* color_mark_array)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total_thread = blockDim.x*gridDim.x;
	for (int i = idx; i < new_rows*new_cols; i += total_thread)
	{
		if (!(new_dst_image_device_ptr[i].r == 0 && new_dst_image_device_ptr[i].g == 0 && new_dst_image_device_ptr[i].b == 0))
			color_mark_array[i] = 1;
		else
			color_mark_array[i] = 0;
	}
}

cudaError_t LaunchCudaCalculateOverlap(const uint8* src, int src_rows, int src_cols, UV_i* new_dst_image_device_ptr, int new_image_rows, int new_image_cols, double& overlap_ratio, int& overlap_pixels)
{
	cudaError_t cudaStatus = cudaSuccess;

	int* src_color_mark_array_cu;
	int* dst_color_mark_array_cu;
	int* overlap_pixels_cu;
	uint8* src_img_cu;
	cudaStatus = cudaMalloc((void**)&src_color_mark_array_cu, src_rows*src_cols * sizeof(int));
	cudaStatus = cudaMalloc((void**)&dst_color_mark_array_cu, new_image_rows*new_image_cols * sizeof(int));
	cudaStatus = cudaMalloc((void**)&overlap_pixels_cu, new_image_rows*new_image_cols * sizeof(int));
	cudaStatus = cudaMalloc((void**)&src_img_cu, src_rows*src_cols * 3 * sizeof(uint8));
	cudaStatus = cudaMemcpy(src_img_cu, src, src_rows*src_cols * 3 * sizeof(uint8), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemset(src_color_mark_array_cu, 0, src_rows*src_cols * sizeof(int));
	cudaStatus = cudaMemset(dst_color_mark_array_cu, 0, new_image_rows*new_image_cols * sizeof(int));

	cuCalculateOverlap << <1024, 512 >> > (src_img_cu, src_rows, src_cols, new_dst_image_device_ptr, new_image_rows, new_image_cols, overlap_pixels_cu);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("cuCalculateOverlap launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("cudaDeviceSynchronize returned error code %d after launching cuCalculateOverlap!\n", cudaStatus);
		return cudaStatus;
	}
	cuMarkColorPixel << <1024, 512 >> > (src_img_cu, src_rows, src_cols, src_color_mark_array_cu);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("cuMarkColorPixel for src_img launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("cudaDeviceSynchronize returned error code %d after launching cuMarkColorPixel for src_img!\n", cudaStatus);
		return cudaStatus;
	}
	cuMarkColorPixelUVI << <1024, 512 >> > (new_dst_image_device_ptr, new_image_rows, new_image_cols, dst_color_mark_array_cu);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("cuMarkColorPixelUVI for dst_img launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("cudaDeviceSynchronize returned error code %d after launching cuMarkColorPixelUVI for dst_img!\n", cudaStatus);
		return cudaStatus;
	}

	thrust::device_ptr<int> row_array_ptr(overlap_pixels_cu);
	int overlap_sum = thrust::reduce(row_array_ptr, row_array_ptr + new_image_rows*new_image_cols);
	thrust::device_ptr<int> src_color_mark_array_ptr(src_color_mark_array_cu);
	int src_color_pixel_num = thrust::reduce(src_color_mark_array_ptr, src_color_mark_array_ptr + src_rows*src_cols);
	thrust::device_ptr<int> dst_color_mark_array_ptr(dst_color_mark_array_cu);
	int dst_color_pixel_num = thrust::reduce(dst_color_mark_array_ptr, dst_color_mark_array_ptr + new_image_rows*new_image_cols);

	overlap_ratio = (double)overlap_sum / (double)std::min<int>(src_color_pixel_num, dst_color_pixel_num);
	overlap_pixels = overlap_sum;


	cudaFree(src_color_mark_array_cu);
	cudaFree(dst_color_mark_array_cu);
	cudaFree(overlap_pixels_cu);
	cudaFree(src_img_cu);

	return cudaStatus;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void cuFusionSrcImage(uint8* src_img_cu, int src_rows, int src_cols, int src_begin_row, int src_begin_col, int src_end_row, int src_end_col, uint8* fusion_img_cu, int fusion_rows, int fusion_cols, int offset_rows, int offset_cols)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total_thread = blockDim.x*gridDim.x;

	int src_valid_rows = src_end_row - src_begin_row;
	int src_valid_cols = src_end_col - src_begin_col;
	for (int i = idx; i < src_valid_rows*src_valid_cols; i += total_thread)
	{
		int src_img_row = i / src_valid_cols + src_begin_row;
		int src_img_col = i%src_valid_cols + src_begin_col;
		int fusion_img_row = src_img_row - offset_rows;
		int fusion_img_col = src_img_col - offset_cols;

		fusion_img_cu[fusion_img_row*fusion_cols * 3 + fusion_img_col * 3] = src_img_cu[src_img_row*src_cols * 3 + src_img_col * 3];
		fusion_img_cu[fusion_img_row*fusion_cols * 3 + fusion_img_col * 3 + 1] = src_img_cu[src_img_row*src_cols * 3 + src_img_col * 3 + 1];
		fusion_img_cu[fusion_img_row*fusion_cols * 3 + fusion_img_col * 3 + 2] = src_img_cu[src_img_row*src_cols * 3 + src_img_col * 3 + 2];
	}
}

__global__ void cuFusionDstImage(UV_i* new_dst_image_device_ptr, int new_image_rows, int new_image_cols, uint8* fusion_img_cu, int fusion_rows, int fusion_cols, int offset_rows, int offset_cols)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total_thread = blockDim.x*gridDim.x;

	for (int i = idx; i<new_image_rows*new_image_cols; i += total_thread)
	{
		int new_dst_row = new_dst_image_device_ptr[i].row_i;
		int new_dst_col = new_dst_image_device_ptr[i].col_i;
		int fusion_img_row = new_dst_row - offset_rows;
		int fusion_img_col = new_dst_col - offset_cols;

		if (!(new_dst_image_device_ptr[i].r == 0 && new_dst_image_device_ptr[i].g == 0 && new_dst_image_device_ptr[i].b == 0))
		{
			fusion_img_cu[fusion_img_row*fusion_cols * 3 + fusion_img_col * 3] = new_dst_image_device_ptr[i].r;
			fusion_img_cu[fusion_img_row*fusion_cols * 3 + fusion_img_col * 3 + 1] = new_dst_image_device_ptr[i].g;
			fusion_img_cu[fusion_img_row*fusion_cols * 3 + fusion_img_col * 3 + 2] = new_dst_image_device_ptr[i].b;
		}
	}
}

cudaError_t LaunchCudaFusionImage(const uint8* src, int src_rows, int src_cols, int src_begin_row, int src_begin_col, int src_end_row, int src_end_col, UV_i* new_dst_image_device_ptr, int new_image_rows, int new_image_cols, uint8* out_fusion_img, int fusion_rows, int fusion_cols, int offset_rows, int offset_cols)
{
	cudaError_t cudaStatus = cudaSuccess;
	uint8* src_img_cu;
	uint8* fusion_img_cu;
	cudaStatus = cudaMalloc((void**)&src_img_cu, src_rows*src_cols * 3 * sizeof(uint8));
	cudaStatus = cudaMalloc((void**)&fusion_img_cu, fusion_rows*fusion_cols * 3 * sizeof(uint8));
	cudaStatus = cudaMemcpy(src_img_cu, src, src_rows*src_cols * 3 * sizeof(uint8), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemset(fusion_img_cu, 0, fusion_rows*fusion_cols * 3 * sizeof(uint8));

	cuFusionSrcImage << <1024, 512 >> > (src_img_cu, src_rows, src_cols, src_begin_row, src_begin_col, src_end_row, src_end_col, fusion_img_cu, fusion_rows, fusion_cols, offset_rows, offset_cols);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("cuFusionSrcImage launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("cudaDeviceSynchronize returned error code %d after launching cuFusionSrcImage!\n", cudaStatus);
		return cudaStatus;
	}
	cuFusionDstImage << <1024, 512 >> > (new_dst_image_device_ptr, new_image_rows, new_image_cols, fusion_img_cu, fusion_rows, fusion_cols, offset_rows, offset_cols);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("cuFusionDstImage launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("cudaDeviceSynchronize returned error code %d after launching cuFusionDstImage!\n", cudaStatus);
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(out_fusion_img, fusion_img_cu, fusion_rows*fusion_cols * 3 * sizeof(uint8), cudaMemcpyDeviceToHost);

	cudaFree(src_img_cu);
	cudaFree(fusion_img_cu);
	cudaFree(new_dst_image_device_ptr);

	return cudaStatus;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////// Only detect intersection /////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void cuOnlyCalculateOverlap(uint8* src, int src_rows, int src_cols, uint8* dst, int dst_rows, int dst_cols, double* affine_mat, int* out_overlap_pixels)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total_thread = blockDim.x*gridDim.x;

	for (int i = idx; i < dst_rows*dst_cols; i += total_thread)
	{
		int dst_row = i / dst_cols;
		int dst_col = i % dst_cols;
		// pixel position
		int dst_r_index = dst_row*dst_cols * 3 + dst_col * 3;
		int dst_g_index = dst_row*dst_cols * 3 + dst_col * 3 + 1;
		int dst_b_index = dst_row*dst_cols * 3 + dst_col * 3 + 2;

		// r,g,b
		uint8 r = dst[dst_r_index];
		uint8 g = dst[dst_g_index];
		uint8 b = dst[dst_b_index];

		if (!(r == 0 && g == 0 && b == 0))
		{
			int transformed_dst_row = (int)(affine_mat[0] * (double)dst_row + affine_mat[1] * (double)dst_col + affine_mat[2]);
			int transformed_dst_col = (int)(affine_mat[3] * (double)dst_row + affine_mat[4] * (double)dst_col + affine_mat[5]);
			if (transformed_dst_row >= 0 && transformed_dst_row<src_rows && transformed_dst_col >= 0 && transformed_dst_col<src_cols)	// locate in the src image.
			{
				int src_r_index = transformed_dst_row*src_cols * 3 + transformed_dst_col * 3;
				int src_g_index = transformed_dst_row*src_cols * 3 + transformed_dst_col * 3 + 1;
				int src_b_index = transformed_dst_row*src_cols * 3 + transformed_dst_col * 3 + 2;
				uint8 src_r = src[src_r_index];
				uint8 src_g = src[src_g_index];
				uint8 src_b = src[src_b_index];
				if (!(src_r == 0 && src_g == 0 && src_b == 0))
				{
					out_overlap_pixels[i] = 1;
				}
			}
		}
	}
}

cudaError_t LaunchCudaOnlyCalculateOverlap(const uint8* src, int src_rows, int src_cols, const uint8* dst, int dst_rows, int dst_cols, double* affine_mat, double& overlap_ratio, int& overlap_pixels)
{
	cudaError_t cudaStatus = cudaSuccess;
	int* src_color_mark_array_cu;
	int* dst_color_mark_array_cu;
	int* overlap_pixels_cu;
	double* affine_mat_cu;
	uint8* src_img_cu;
	uint8* dst_img_cu;
	cudaStatus = cudaMalloc((void**)&src_color_mark_array_cu, src_rows*src_cols * sizeof(int));
	cudaStatus = cudaMalloc((void**)&dst_color_mark_array_cu, dst_rows*dst_cols * sizeof(int));
	cudaStatus = cudaMalloc((void**)&overlap_pixels_cu, dst_rows*dst_cols * sizeof(int));
	cudaStatus = cudaMalloc((void**)&affine_mat_cu, 6 * sizeof(double));
	cudaStatus = cudaMalloc((void**)&src_img_cu, src_rows*src_cols * 3 * sizeof(uint8));
	cudaStatus = cudaMalloc((void**)&dst_img_cu, dst_rows*dst_cols * 3 * sizeof(uint8));
	cudaStatus = cudaMemcpy(src_img_cu, src, src_rows*src_cols * 3 * sizeof(uint8), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(dst_img_cu, dst, dst_rows*dst_cols * 3 * sizeof(uint8), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(affine_mat_cu, affine_mat, 6 * sizeof(double), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemset(overlap_pixels_cu, 0, dst_rows*dst_cols * sizeof(int));
	cudaStatus = cudaMemset(src_color_mark_array_cu, 0, src_rows*src_cols * sizeof(int));
	cudaStatus = cudaMemset(dst_color_mark_array_cu, 0, dst_rows*dst_cols * sizeof(int));

	cuOnlyCalculateOverlap << <1024, 512 >> > (src_img_cu, src_rows, src_cols, dst_img_cu, dst_rows, dst_cols, affine_mat_cu, overlap_pixels_cu);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("cuOnlyCalculateOverlap launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("cudaDeviceSynchronize returned error code %d after launching cuOnlyCalculateOverlap!\n", cudaStatus);
		return cudaStatus;
	}
	cuMarkColorPixel << <1024, 512 >> > (src_img_cu, src_rows, src_cols, src_color_mark_array_cu);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("cuMarkColorPixel for src_img launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("cudaDeviceSynchronize returned error code %d after launching cuMarkColorPixel for src_img!\n", cudaStatus);
		return cudaStatus;
	}
	cuMarkColorPixel << <1024, 512 >> > (dst_img_cu, dst_rows, dst_cols, dst_color_mark_array_cu);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("cuMarkColorPixel for dst_img launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("cudaDeviceSynchronize returned error code %d after launching cuMarkColorPixel for dst_img!\n", cudaStatus);
		return cudaStatus;
	}


	thrust::device_ptr<int> row_array_ptr(overlap_pixels_cu);
	int overlap_sum = thrust::reduce(row_array_ptr, row_array_ptr + dst_rows*dst_cols);
	thrust::device_ptr<int> src_color_mark_array_ptr(src_color_mark_array_cu);
	int src_color_pixel_num = thrust::reduce(src_color_mark_array_ptr, src_color_mark_array_ptr + src_rows*src_cols);
	thrust::device_ptr<int> dst_color_mark_array_ptr(dst_color_mark_array_cu);
	int dst_color_pixel_num = thrust::reduce(dst_color_mark_array_ptr, dst_color_mark_array_ptr + dst_rows*dst_cols);

	overlap_ratio = (double)overlap_sum / (double)std::min<int>(src_color_pixel_num, dst_color_pixel_num);
	overlap_pixels = overlap_sum;


	cudaFree(overlap_pixels_cu);
	cudaFree(src_color_mark_array_cu);
	cudaFree(dst_color_mark_array_cu);
	cudaFree(affine_mat_cu);
	cudaFree(src_img_cu);
	cudaFree(dst_img_cu);

	return cudaStatus;
}




//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////   Initialize all of fragments into GPU memoery, and then detect intersection given a series of poses  ////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
cudaError_t LaunchCudaInitializeGPUFragments(const std::vector<uint8*>& cpu_fragments, const std::vector<int>& fragment_rows, const std::vector<int>& fragment_cols, std::vector<uint8*>& out_fragment_gpu_ptrs)
{
	cudaError_t cudaStatus = cudaSuccess;
	for (int i = 0; i<cpu_fragments.size(); ++i)
	{
		uint8* img_cpu = cpu_fragments[i];
		int rows = fragment_rows[i];
		int cols = fragment_cols[i];
		uint8* img_gpu;
		cudaStatus = cudaMalloc((void**)&img_gpu, rows*cols * 3 * sizeof(uint8));
		cudaStatus = cudaMemcpy(img_gpu, img_cpu, rows*cols * 3 * sizeof(uint8), cudaMemcpyHostToDevice);
		out_fragment_gpu_ptrs.push_back(img_gpu);
	}
	return cudaStatus;
}



cudaError_t LaunchCudaDetectBatchPosesIntersection(const std::vector<IdAndPose>& cpu_poses1, const std::vector<IdAndPose>& cpu_poses2, const std::vector<uint8*>& gpu_fragments, const std::vector<int>& fragment_rows, const std::vector<int>& fragment_cols, int& out_overlap_pixels, std::vector<int>& out_pose2_overlap_pixel_array)
{
	cudaError_t cudaStatus = cudaSuccess;

	std::vector<int*> overlap_pixels_cu(cpu_poses2.size());
	std::vector<double*> relative_transform_cu(cpu_poses1.size()*cpu_poses2.size());
	for (int i = 0; i<cpu_poses2.size(); ++i)
	{
		int pose2_id = cpu_poses2[i].fragmentId;
		cudaStatus = cudaMalloc((void**)&overlap_pixels_cu[i], fragment_rows[pose2_id] * fragment_cols[pose2_id] * sizeof(int));
		cudaStatus = cudaMemset(overlap_pixels_cu[i], 0, fragment_rows[pose2_id] * fragment_cols[pose2_id] * sizeof(int));
		for (int j = 0; j<cpu_poses1.size(); ++j)
		{
			// pose1^-1 * pose2
			double* pose1_inv = InverseMat(cpu_poses1[j].pose);
			double* relative = MatMul(pose1_inv, cpu_poses2[i].pose);
			cudaStatus = cudaMalloc((void**)&relative_transform_cu[i*cpu_poses1.size() + j], 6 * sizeof(double));
			cudaStatus = cudaMemcpy(relative_transform_cu[i*cpu_poses1.size() + j], relative, 6 * sizeof(double), cudaMemcpyHostToDevice);
			delete[] pose1_inv;
			delete[] relative;
		}
	}

	for (int i = 0; i < cpu_poses2.size(); ++i)
	{
		for (int j = 0; j < cpu_poses1.size(); ++j)
		{
			int pose1_id = cpu_poses1[j].fragmentId;
			int pose2_id = cpu_poses2[i].fragmentId;
			cuOnlyCalculateOverlap << <256, 512 >> > (gpu_fragments[pose1_id], fragment_rows[pose1_id], fragment_cols[pose1_id], gpu_fragments[pose2_id], fragment_rows[pose2_id], fragment_cols[pose2_id], relative_transform_cu[i*cpu_poses1.size() + j], overlap_pixels_cu[i]);
		}
	}
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("cuOnlyCalculateOverlap in batch pose launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("cudaDeviceSynchronize returned error code %d after launching cuOnlyCalculateOverlap in batch pose\n", cudaStatus);
		return cudaStatus;
	}

	int overlap_sum = 0;
	for (int i = 0; i<overlap_pixels_cu.size(); ++i)
	{
		int pose2_id = cpu_poses2[i].fragmentId;
		int rows = fragment_rows[pose2_id];
		int cols = fragment_cols[pose2_id];
		thrust::device_ptr<int> row_array_ptr(overlap_pixels_cu[i]);
		int overlap_pixel_num = thrust::reduce(row_array_ptr, row_array_ptr + rows*cols);
		out_pose2_overlap_pixel_array.push_back(overlap_pixel_num);
		overlap_sum += overlap_pixel_num;
	}
	out_overlap_pixels = overlap_sum;

	for (auto ptr_cu : overlap_pixels_cu)
		cudaFree(ptr_cu);
	for (auto ptr_cu : relative_transform_cu)
		cudaFree(ptr_cu);

	return cudaStatus;
}
