#include <vector>
#include <algorithm> 
#include <omp.h>

#include "TransformImage.cuh"
#include "TransformImage.h"
#include "Utils.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////// Fusion two images and detect intersection //////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// CPU
std::vector<UV_i> FusionBoundingBox(const std::vector<UV_i>& boundingbox1, const std::vector<UV_i>& boundingbox2)
{
	std::vector<int> ro = { boundingbox1[0].row_i, boundingbox1[1].row_i, boundingbox2[0].row_i, boundingbox2[1].row_i };
	std::vector<int> co = { boundingbox1[0].col_i, boundingbox1[1].col_i, boundingbox2[0].col_i, boundingbox2[1].col_i };

	int new_min_row = *std::min_element(ro.begin(), ro.end());
	int new_min_col = *std::min_element(co.begin(), co.end());
	int new_max_row = *std::max_element(ro.begin(), ro.end());
	int new_max_col = *std::max_element(co.begin(), co.end());

	std::vector<UV_i> new_boundingbox;
	UV_i min_px, max_px;
	min_px.row_i = new_min_row;
	min_px.col_i = new_min_col;
	max_px.row_i = new_max_row;
	max_px.col_i = new_max_col;
	new_boundingbox.push_back(min_px);
	new_boundingbox.push_back(max_px);

	return new_boundingbox;
}


std::vector<UV_i> FindBoundingBox(uint8* img, int img_rows, int img_cols, double* affine_mat)
{
#define NUM_THREADS 8
	std::vector<int> min_rows(NUM_THREADS, 99999), min_cols(NUM_THREADS, 99999), max_rows(NUM_THREADS, -99999), max_cols(NUM_THREADS, -99999);

	if (affine_mat == NULL)
	{
#pragma omp parallel for num_threads(NUM_THREADS)
		for (int i = 0; i<img_rows; ++i)
		{
			for (int j = 0; j<img_cols; ++j)
			{
				uint8 r = img[i*img_cols * 3 + j * 3];
				uint8 g = img[i*img_cols * 3 + j * 3 + 1];
				uint8 b = img[i*img_cols * 3 + j * 3 + 2];
				if (!(r == 0 && g == 0 && b == 0))
				{
					int id = omp_get_thread_num();
					if (min_rows[id] > i)
						min_rows[id] = i;
					if (min_cols[id] > j)
						min_cols[id] = j;
					if (max_rows[id] < i)
						max_rows[id] = i;
					if (max_cols[id] < j)
						max_cols[id] = j;
				}
			}
		}
	}
	else
	{
#pragma omp parallel for num_threads(NUM_THREADS)
		for (int i = 0; i<img_rows; ++i)
		{
			for (int j = 0; j<img_cols; ++j)
			{
				uint8 r = img[i*img_cols * 3 + j * 3];
				uint8 g = img[i*img_cols * 3 + j * 3 + 1];
				uint8 b = img[i*img_cols * 3 + j * 3 + 2];
				if (!(r == 0 && g == 0 && b == 0))
				{
					int new_row = (int)(affine_mat[0] * (double)i + affine_mat[1] * (double)j + affine_mat[2]);
					int new_col = (int)(affine_mat[3] * (double)i + affine_mat[4] * (double)j + affine_mat[5]);

					int id = omp_get_thread_num();
					if (min_rows[id] > new_row)
						min_rows[id] = new_row;
					if (min_cols[id] > new_col)
						min_cols[id] = new_col;
					if (max_rows[id] < new_row)
						max_rows[id] = new_row;
					if (max_cols[id] < new_col)
						max_cols[id] = new_col;
				}
			}
		}
	}

	int min_row = *std::min_element(min_rows.begin(), min_rows.end());
	int min_col = *std::min_element(min_cols.begin(), min_cols.end());
	int max_row = *std::max_element(max_rows.begin(), max_rows.end());
	int max_col = *std::max_element(max_cols.begin(), max_cols.end());
	UV_i min_p, max_p;
	min_p.row_i = min_row;
	min_p.col_i = min_col;
	max_p.row_i = max_row + 1;
	max_p.col_i = max_col + 1;
	std::vector<UV_i> boundingbox = { min_p, max_p };

	return boundingbox;
}


// dll main function
void InitCUDA()
{
	cuInit();
}
void CloseCUDA()
{
	cuDestory();
}

/*
* Parameters
* src: source image (RGB) 1-d array size will be src_rows*src_cols*3, e.g. [255, 255, 255, 0, 0, 0] is white pixel+black pixel
* dst: denstiny image (RGB) 1-d array. Same with src
* affine_mat: transform matrix. 1-d array, length should be 6 [r11, r12, t1, r21, r22, t2]. Will transform dst image to src image (dst->src).
* out_rows: the number of row in fusioned image
* out_cols: the number of col in fusioned image
* out_overlap_ratio: overlap ratio after apply transform
* out_offset_row, outoffset_col: the source image is moved the offset pixel to new image.
* out_overlap_pixels: the number of overlapped pixels.
*
* Return
* the fusioned image (RGB) . Format is same with src, dst.
* Or NULL if something wrong
*/
uint8* TransformImage(uint8* src, int src_rows, int src_cols, uint8* dst, int dst_rows, int dst_cols, double* affine_mat, int& out_rows, int& out_cols, double& out_overlap_ratio, int& out_offset_row, int& out_offset_col, int& out_overlap_pixels)
{
	uint8* fusioned_image = NULL;
	cudaError_t cuda_error = cudaSuccess;

	// find bounding box of src, dst image, GPU
	std::vector<UV_i> src_boundingbox, dst_boundingbox;
	//cuda_error = LaunchCudaFindBoundingBox(src, src_rows, src_cols, src_boundingbox, NULL);
	//cuda_error = LaunchCudaFindBoundingBox(dst, dst_rows, dst_cols, dst_boundingbox, affine_mat);		// get dst bounding box after transform
	src_boundingbox = FindBoundingBox(src, src_rows, src_cols, NULL);
	dst_boundingbox = FindBoundingBox(dst, dst_rows, dst_cols, affine_mat);

	// dst transformed bounding box
	if (src_boundingbox[0].row_i == -999 || dst_boundingbox[0].row_i == -999)
		return NULL;
	std::vector<UV_i> new_boundingbox = dst_boundingbox;

	// remap to fill up transformed dst bounding box, GPU
	UV_i new_dst_start_pt = new_boundingbox[0];
	UV_i new_dst_end_pt = new_boundingbox[1];
	double* remap_transform = InverseMat(affine_mat);

	UV_i* out_new_dst_image_device_ptr = NULL;
	cuda_error = LaunchCudaRemapFillNewImage(dst, dst_rows, dst_cols, new_dst_start_pt, new_dst_end_pt, remap_transform, out_new_dst_image_device_ptr);
	if (!out_new_dst_image_device_ptr)
		return NULL;

	// fusion 
	int new_rows = new_dst_end_pt.row_i - new_dst_start_pt.row_i;
	int new_cols = new_dst_end_pt.col_i - new_dst_start_pt.col_i;
	// 1. calculate overlap
	double overlap_ratio = 0;
	int overlap_pixels = 0;
	cuda_error = LaunchCudaCalculateOverlap(src, src_rows, src_cols, out_new_dst_image_device_ptr, new_rows, new_cols, overlap_ratio, overlap_pixels);
	out_overlap_ratio = overlap_ratio;
	out_overlap_pixels = overlap_pixels;
	// 2. get the fusion image bounding box
	std::vector<UV_i> fusioned_boundingbox;
	fusioned_boundingbox = FusionBoundingBox(src_boundingbox, new_boundingbox);
	int fusion_row = fusioned_boundingbox[1].row_i - fusioned_boundingbox[0].row_i;
	int fusion_col = fusioned_boundingbox[1].col_i - fusioned_boundingbox[0].col_i;
	int offset_row = fusioned_boundingbox[0].row_i;
	int offset_col = fusioned_boundingbox[0].col_i;

	int src_begin_row = src_boundingbox[0].row_i;
	int src_begin_col = src_boundingbox[0].col_i;
	int src_end_row = src_boundingbox[1].row_i;
	int src_end_col = src_boundingbox[1].col_i;
	fusioned_image = new uint8[fusion_row*fusion_col * 3];
	cuda_error = LaunchCudaFusionImage(src, src_rows, src_cols, src_begin_row, src_begin_col, src_end_row, src_end_col, out_new_dst_image_device_ptr, new_rows, new_cols, fusioned_image, fusion_row, fusion_col, offset_row, offset_col);
	out_rows = fusion_row;
	out_cols = fusion_col;
	out_offset_row = -offset_row;
	out_offset_col = -offset_col;

	delete[] remap_transform;
	return fusioned_image;
}



void DeleteFusionImage(uint8* fusion_image)
{
	delete[] fusion_image;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////// Only detect intersection /////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
*Purely detect intersection. No image fusion.
*
* Parameters:
* src: source image (RGB) 1-d array size will be src_rows*src_cols*3, e.g. [255, 255, 255, 0, 0, 0] is white pixel+black pixel
* dst: denstiny image (RGB) 1-d array. Same with src
* affine_mat: transform matrix. 1-d array, length should be 6 [r11, r12, t1, r21, r22, t2]. Will transform dst image to src image (dst->src).
* out_overlap_pixels: the number of overlapped pixels.
* out_overlap_ratio: overlap ratio after apply transform
*
*/
void OnlyCalculateIntersection(uint8* src, int src_rows, int src_cols, uint8* dst, int dst_rows, int dst_cols, double* affine_mat, int& out_overlap_pixels, double& out_overlap_ratio)
{
	double* remap_transform = InverseMat(affine_mat);
	cudaError_t cuda_error = cudaSuccess;
	double overlap_ratio = 0;
	int overlap_pixels = 0;
	if (src_rows*src_cols<dst_rows*dst_cols)		// because src < dst, exchange src and dst to speed calculating
		cuda_error = LaunchCudaOnlyCalculateOverlap(dst, dst_rows, dst_cols, src, src_rows, src_cols, remap_transform, overlap_ratio, overlap_pixels);
	else
		cuda_error = LaunchCudaOnlyCalculateOverlap(src, src_rows, src_cols, dst, dst_rows, dst_cols, affine_mat, overlap_ratio, overlap_pixels);
	out_overlap_pixels = overlap_pixels;
	out_overlap_ratio = overlap_ratio;
	delete[] remap_transform;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////   Initialize all of fragments into GPU memoery, and then detect intersection given a series of poses  ////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
* initialize all of fragment images into GPU memory.
* Parameters
* cpu_fragments.size == fragment_rows.size == fragment_cols.size; They should be matched.
*/
std::vector<uint8*> InitializeGPUFragments(const std::vector<uint8*>& cpu_fragments, const std::vector<int>& fragment_rows, const std::vector<int>& fragment_cols)
{
	std::vector<uint8*> out_fragment_gpu_ptrs;
	cudaError_t cuda_error = cudaSuccess;
	cuda_error = LaunchCudaInitializeGPUFragments(cpu_fragments, fragment_rows, fragment_cols, out_fragment_gpu_ptrs);

	return out_fragment_gpu_ptrs;
}

/*
* Given two batch of poses and all of gpu image fragments, detect intersection
* Parameters
* cpu_poses1: the first batch of poses and corresponding fragmentId. e.g. [(0, pose0), (2, pose2)]
* cpu_poses2: the second batch of poses and corresponding fragmentId. e.g. [(1, pose1), (3, pose3)]
* gpu_fragments: all of image fragments (gpu pointer)
* fragment_rows: fragment's rows. e.g. fragment_rows[2] is the rows of image gpu_fragments[2]
* fragment_cols: fragment;s cols
* out_overlap_pixels: total overlap pixels = sum(out_pose2_overlap_pixel_array)
* out_pose2_overlap_pixel_array: the number of overlapped pixels in the second batch. e.g. out_pose2_overlap_pixel_array[1] means the number of overlapped pixels in fragment cpu_poses2[1]
*/
void DetectBatchPosesIntersection(const std::vector<IdAndPose>& cpu_poses1, const std::vector<IdAndPose>& cpu_poses2, const std::vector<uint8*>& gpu_fragments, const std::vector<int>& fragment_rows, const std::vector<int>& fragment_cols, int& out_overlap_pixels, std::vector<int>& out_pose2_overlap_pixel_array)
{
	cudaError_t cuda_error = cudaSuccess;
	cuda_error = LaunchCudaDetectBatchPosesIntersection(cpu_poses1, cpu_poses2, gpu_fragments, fragment_rows, fragment_cols, out_overlap_pixels, out_pose2_overlap_pixel_array);
}

