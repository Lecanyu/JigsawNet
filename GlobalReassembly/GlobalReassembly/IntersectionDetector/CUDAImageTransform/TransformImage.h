#pragma once
#include "Utils.h"

extern "C" __declspec(dllexport) void InitCUDA();
extern "C" __declspec(dllexport) void CloseCUDA();
extern "C" __declspec(dllexport) uint8* TransformImage(uint8* src, int src_rows, int src_cols, uint8* dst, int dst_rows, int dst_cols, double* affine_mat, int& out_rows, int& out_cols, double& out_overlap_ratio, int& offset_row, int& offset_col, int& out_overlap_pixels);
extern "C" __declspec(dllexport) void DeleteFusionImage(uint8* fusion_image);

extern "C" __declspec(dllexport) void OnlyCalculateIntersection(uint8* src, int src_rows, int src_cols, uint8* dst, int dst_rows, int dst_cols, double* affine_mat, int& out_overlap_pixels, double& out_overlap_ratio);


std::vector<uint8*> InitializeGPUFragments(const std::vector<uint8*>& cpu_fragments, const std::vector<int>& fragment_rows, const std::vector<int>& fragment_cols);
void DetectBatchPosesIntersection(const std::vector<IdAndPose>& cpu_poses1, const std::vector<IdAndPose>& cpu_poses2, const std::vector<uint8*>& gpu_fragments, const std::vector<int>& fragment_rows, const std::vector<int>& fragment_cols, int& out_overlap_pixels, std::vector<int>& out_pose2_overlap_pixel_array);

