#ifndef TransformImageCUDA_H
#define TransformImageCUDA_H

#include "TransformImage.h"
#include "cuda_runtime.h"

cudaError_t cuInit();
cudaError_t cuDestory();
cudaError_t LaunchCudaFindBoundingBox(const uint8* image, int rows, int cols, std::vector<UV_i>& out_boundingbox, double* affine = NULL);
cudaError_t LaunchCudaRemapFillNewImage(const uint8* before_trans_image, int rows, int cols, UV_i new_dst_start_pt, UV_i new_dst_end_pt, double* affine_mat, UV_i*& out_new_dst_image_device_ptr);
cudaError_t LaunchCudaCalculateOverlap(const uint8* src, int src_rows, int src_cols, UV_i* out_new_dst_image_device_ptr, int new_image_rows, int new_image_cols, double& overlap_ratio, int& overlap_pixels);
cudaError_t LaunchCudaFusionImage(const uint8* src, int src_rows, int src_cols, int src_begin_row, int src_begin_col, int src_end_row, int src_end_col, UV_i* new_dst_image_device_ptr, int new_image_rows, int new_image_cols, uint8* out_fusion_img, int fusion_rows, int fusion_cols, int offset_rows, int offset_cols);


cudaError_t LaunchCudaOnlyCalculateOverlap(const uint8* src, int src_rows, int src_cols, const uint8* dst, int dst_rows, int dst_cols, double* affine_mat, double& overlap_ratio, int& overlap_pixels);


cudaError_t LaunchCudaInitializeGPUFragments(const std::vector<uint8*>& cpu_fragments, const std::vector<int>& fragment_rows, const std::vector<int>& fragment_cols, std::vector<uint8*>& out_fragment_gpu_ptrs);
cudaError_t LaunchCudaDetectBatchPosesIntersection(const std::vector<IdAndPose>& cpu_poses1, const std::vector<IdAndPose>& cpu_poses2, const std::vector<uint8*>& gpu_fragments, const std::vector<int>& fragment_rows, const std::vector<int>& fragment_cols, int& out_overlap_pixels, std::vector<int>& out_pose2_overlap_pixel_array);



#endif
