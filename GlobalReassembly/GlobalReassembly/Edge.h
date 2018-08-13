#pragma once

#include <Eigen/Eigen>
#include <Eigen/Core>

class Edge
{
public:
	int frame1_;			// one vertex id connected by edge
	int frame2_;			// the other vertex id
	double score_;			// alignment raw score
	int rank_;				// score rank in the edges which they connect the same two vertices 
	Eigen::Matrix3d transformation_;	// alignment transformation
	bool select_;			// select mark

	int idReflect_;			// this edge id in the vector array 

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	Edge()
		:frame1_(-1), frame2_(-1), score_(0.0),transformation_(Eigen::Matrix3d::Identity()), select_(false), idReflect_(-1), rank_(-1)
	{}

	Edge(int frame1, int frame2, double score, const Eigen::Matrix3d& trans, int idReflect, bool select = false)
		:frame1_(frame1), frame2_(frame2), score_(score), transformation_(trans), idReflect_(idReflect), select_(select)
	{}

};

class SortEdge
{
public:
	bool operator()(const Edge& e1, const Edge& e2)
	{
		return e1.score_ > e2.score_;
	}
};