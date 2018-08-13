#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Utils.h"
#include "Edge.h"
#include "Parameters.h"
#include "Loop.h"

class MultiGraph
{
public:
	int vertexNum_;
	std::vector<Edge, Eigen::aligned_allocator<Edge>> edges_;
	std::unordered_map<IdPair, MultiEdgeId, hash_func, key_equal> mapIdPair2EdgeIds_;
	std::unordered_map<IdRank, int, IdRank_hash_func, IdRank_key_equal> mapIdRank2EdgeIds_;

	// init
	MultiGraph(Alignments2d& alignments);
	MultiGraph(std::string align_file, int align_file_type);

	MultiEdgeId& operator()(int i, int j);
	Edge& operator[](const int i) { return edges_[i]; }

	// Judge connected by UFS
	bool UnionFindSetJudgeAllLink();

	// Loop generator.
	// Instead of DFS search, here we 1. Generator a combination; 2. validate this combination;
	// It is NP-hard problem, we only implement length=3 and 4.
	std::experimental::generator<std::vector<int>> LoopGenerator(int loop_length);

	// A greedy strategy to generate loop
	std::experimental::generator<std::vector<int>> LoopGeneratorGreedy(int loop_length);


	// Print function for debug.
	void print(int id1, int id2)
	{
		std::cout << "**********************************************\n";
		std::cout << "frame " << id1 << "----" << id2 << "\n";
		for (int i = 0; i<edges_.size(); ++i)
		{
			if (edges_[i].frame1_ == id1 && edges_[i].frame2_ == id2)
			{
				std::cout << "score: " << edges_[i].score_ << "\n";
				std::cout << edges_[i].transformation_ << "\n";
			}
		}
	}
	void print(int edgeId)
	{
		std::cout << "**********************************************\n";
		std::cout << "frame " << edges_[edgeId].frame1_ << "----" << edges_[edgeId].frame2_ << "\n";
		std::cout << "score: " << edges_[edgeId].score_ << ", rank: "<<edges_[edgeId].rank_<<"\n";
		std::cout << edges_[edgeId].transformation_ << "\n";
	}
private:

	// NULL object for checking return
	MultiEdgeId emptyMultiEdgeObj_;
	// Judge generated loop connected or not.
	bool LoopConnected(const std::vector<int>& loop);
};
