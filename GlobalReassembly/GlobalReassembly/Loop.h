#pragma once

#include <Eigen/Eigen>
#include <Eigen/Core>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "IntersectionDetector/CUDAImage.h"

struct LoopEdge
{
	int frame1_, frame2_, rank_;
	LoopEdge():frame1_(-1), frame2_(-1),rank_(-1){}
	LoopEdge(int v1, int v2, int rank):frame1_(v1),frame2_(v2),rank_(rank){}
};

// when number of frames < 10000 and rank < 1000 , there is no conflict.
struct HashLoopEdge
{
	int operator()(const LoopEdge& l) const
	{
		return l.frame1_*1e7 + l.frame2_ * 1e3 + l.rank_;
	}
};

struct LoopEdgeEq
{
	bool operator()(const LoopEdge& l1, const LoopEdge& l2) const
	{
		return l1.frame1_ == l2.frame1_ && l1.frame2_ == l2.frame2_ && l1.rank_ == l2.rank_;
	}
};



class LoopClosure
{
public:
	std::unordered_set<int> vertices_;
	std::unordered_set<LoopEdge, HashLoopEdge, LoopEdgeEq> edges_;
	std::unordered_map<int, Eigen::Matrix3d> poses_;
	std::unordered_map<int, int> countPixels_;
	double probs_;

	LoopClosure():probs_(-999.0){};
	LoopClosure(std::unordered_set<int>& vs, std::unordered_set<LoopEdge, HashLoopEdge, LoopEdgeEq>& es, std::unordered_map<int, Eigen::Matrix3d>& poses, std::unordered_map<int, int>& pixels_num, double probs)
		:vertices_(vs), edges_(es), poses_(poses), countPixels_(pixels_num), probs_(probs)
	{};


	// if two edge set are the same, then the two loop closure are the same.
	bool operator==(const LoopClosure& loop_closure) const
	{
		if (edges_.size() == loop_closure.edges_.size())
		{
			for (const LoopEdge& edge : edges_)
			{
				if (loop_closure.edges_.find(edge) == loop_closure.edges_.end())
					return false;
			}
			return true;
		}
		else
			return false;
	}
};


struct SortLoopClosure
{
	bool operator()(const LoopClosure& loop1, const LoopClosure& loop2)
	{
		return loop1.probs_ > loop2.probs_;
	}
};