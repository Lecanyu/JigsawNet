#pragma once
#include "Parameters.h"
#include "MultiGraph.h"
#include "Loop.h"
#include "IntersectionDetector/CUDAImgIntersectionDetector.h"

class JigsawOpt2
{
public:
	// core variable
	MultiGraph multiGraph_;
	Alignments2d pose_;
	std::unordered_map<int, std::vector<int>> poseChunk_;

	std::vector<cv::Mat> all_fragment_images;

	JigsawOpt2(std::string align_file, int align_file_type) :multiGraph_(align_file, align_file_type)
	{
		for (int i = 0; i < multiGraph_.vertexNum_; ++i)
		{
			FramedTransformation2d trans(i, i, 0.0, Eigen::Matrix3d::Identity());
			pose_.data_.push_back(trans);
		}
		pose_.frame_num_ = multiGraph_.vertexNum_;
	}
	JigsawOpt2(Alignments2d& alignment) :multiGraph_(alignment)
	{
		for (int i = 0; i < multiGraph_.vertexNum_; ++i)
		{
			FramedTransformation2d trans(i, i, 0.0, Eigen::Matrix3d::Identity());
			pose_.data_.push_back(trans);
		}
		pose_.frame_num_ = multiGraph_.vertexNum_;
	}
	// progressive opt based on loop + intersection detection
	std::unordered_map<int, cv::Mat> OptWithInterSec(const std::vector<CUDAImage>& original_images, const std::vector<uint8*>& gpu_fragments, const std::vector<int>& fragment_rows, const std::vector<int>& fragment_cols);

	// Output selected edge
	void OutputSelectEdge(std::string filename);

private:
	// deep-first-search to calculate all of vertices' pose
	void DFSPose(const std::vector<int>& vertex2set);

	// generate one loop for validating
	// In: abstract_loop, such as 1,2,3
	// Out: concrete_loop array, i.e. edge indices 
	std::vector<std::vector<int>> GenerateValidableLoop(std::vector<int> abstract_loop, const int beam_width);

	// recursive generator single concrete loop
	void RecurLoop(const int beam_width, const std::vector<int> abstract_loop, int level, std::vector<int> loop_candidate, std::vector<std::vector<int>>& out_result);

	// Check error
	void CalculateError2d(const std::vector<int>& abstract_loop, const std::vector<int>& concrete_loop, double& out_translation_err, double& out_rotation_err);
	bool IsSameMatrix(const Eigen::Matrix3d& mat1, const Eigen::Matrix3d& mat2);
	// Condition1
	// no common vertices should not be verlapped
	bool Condition1(LoopClosure& loop1, LoopClosure& loop2, const std::vector<uint8*>& gpu_fragments, const std::vector<int>& fragment_rows, const std::vector<int>& fragment_cols);

	// Condition 2
	// common vertices should be the same poses
	bool Condition2(LoopClosure& loop1, LoopClosure& loop2);

	// Condition 3
	// if common vertices are the same poses and some new vertices in loop2, return true
	// else return false
	bool Condition3(LoopClosure& loop1, LoopClosure& loop2);

	// Initialize all of smallest loop closures
	void InitializeSmallestLoopClosure(std::vector<LoopClosure>& loop_closures, const std::vector<CUDAImage>& original_images);
	// Bottom to top merging
	void Bottom2TopMerge(std::vector<LoopClosure>& loop_closures, std::vector<std::vector<LoopClosure>>& loop_closure_history, const std::vector<uint8*>& gpu_fragments, const std::vector<int>& fragment_rows, const std::vector<int>& fragment_cols);
	// Top to bottom merging
	void Top2BottomMerge(LoopClosure& best_loop, std::vector<std::vector<LoopClosure>>& loop_closure_history, const std::vector<uint8*>& gpu_fragments, const std::vector<int>& fragment_rows, const std::vector<int>& fragment_cols);

	// debug show loop closure
	void ShowLoopClosures(const std::vector<LoopClosure>& loop_closures);
};

