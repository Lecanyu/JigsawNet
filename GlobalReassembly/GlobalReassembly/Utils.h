#pragma once

#include <vector>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Eigen>
#include <experimental/generator>

//////////////////////////////  Loop ////////////////////////////////////////////
struct Loop
{
	std::vector<int> vertices;
	double score;
	Loop():score(0.0){}
	Loop(const std::vector<int>& v, double s):vertices(v),score(s){}
};

struct LoopSort
{
	bool operator()(const Loop& l1, const Loop& l2)
	{
		return l1.score > l2.score;
	}
};

//////////////////////////////  File IO ////////////////////////////////////////////
struct FramedTransformation2d {
	int frame1_;
	int frame2_;
	double score_;

	Eigen::Matrix3d transformation_;

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	FramedTransformation2d(){}
	FramedTransformation2d(int frame1, int frame2, double score, const Eigen::Matrix3d& t)
		: frame1_(frame1), frame2_(frame2), score_(score), transformation_(t)
	{}
};

struct SortFramedTransformation2d
{
	bool operator () (const FramedTransformation2d &k1, const FramedTransformation2d &k2)
	{
		// check frame1
		if (k1.frame1_ < k2.frame1_)
			return true;
		else if (k1.frame1_ == k2.frame1_) {
			// check frame2
			if (k1.frame2_ < k2.frame2_)
				return true;
			else if (k1.frame2_ == k2.frame2_) {
				// check score
				if (k1.score_ > k2.score_)
					return true;
				else
					return false;
			}
			else
				return false;
		}
		else
			return false;
	}
};

struct SortScore
{
	bool operator () (const FramedTransformation2d &k1, const FramedTransformation2d &k2)
	{
		return k1.score_ > k2.score_;
	}
};

struct Alignments2d {
	std::vector< FramedTransformation2d, Eigen::aligned_allocator<FramedTransformation2d>> data_;
	// total frame numbers
	int frame_num_;

	void LoadFromFile(std::string filename) {
		data_.clear();
		int frame1, frame2;
		double score;
		Eigen::Matrix3d trans;
		char temp[128];
		int n;

		FILE * f = fopen(filename.c_str(), "r");
		if (f != NULL) {
			char buffer[102400];
			while (fgets(buffer, 102400, f) != NULL) {
				if (strlen(buffer) > 0 && buffer[0] != '#') {
					sscanf(buffer, "%s %d", temp, &n);
					std::string head = temp;
					if (head == "Node")
					{
						frame_num_ = n + 1;
						continue;
					}
					else
					{
						sscanf(buffer, "%d %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", &frame1, &frame2, &score,
							&trans(0, 0), &trans(0, 1), &trans(0, 2),
							&trans(1, 0), &trans(1, 1), &trans(1, 2),
							&trans(2, 0), &trans(2, 1), &trans(2, 2));
						data_.push_back(FramedTransformation2d(frame1, frame2, score, trans));
					}
				}
			}
			fclose(f);
		}
	}
	void LoadFromPairwiseMartix(std::string filename)
	{
		data_.clear();
		frame_num_ = -1;
		int frame1, frame2;
		double score;
		Eigen::Matrix3d trans;
		FILE * f = fopen(filename.c_str(), "r");
		if (f != NULL) {
			char buffer[1024];
			while (fgets(buffer, 1024, f) != NULL) {
				if (strlen(buffer) > 0 && buffer[0] != '#') {
					sscanf(buffer, "%d %d %lf", &frame1, &frame2, &score);
					fgets(buffer, 1024, f);
					sscanf(buffer, "%lf %lf %lf", &trans(0, 0), &trans(0, 1), &trans(0, 2));
					fgets(buffer, 1024, f);
					sscanf(buffer, "%lf %lf %lf", &trans(1, 0), &trans(1, 1), &trans(1, 2));
					fgets(buffer, 1024, f);
					sscanf(buffer, "%lf %lf %lf", &trans(2, 0), &trans(2, 1), &trans(2, 2));

					data_.push_back(FramedTransformation2d(frame1, frame2, score, trans));
					if (frame_num_ < std::max<int>(frame1, frame2))
						frame_num_ = std::max<int>(frame1, frame2);
				}
			}
			fclose(f);
			frame_num_ += 1;
		}
	}

	void LoadFromPose(std::string filename)
	{
		data_.clear();
		int frame1, frame2;
		Eigen::Matrix3d trans;
		FILE * f = fopen(filename.c_str(), "r");
		if (f != NULL) {
			char buffer[1024];
			while (fgets(buffer, 1024, f) != NULL) {
				if (strlen(buffer) > 0 && buffer[0] != '#') {
					sscanf(buffer, "%d %d", &frame1, &frame2);
					fgets(buffer, 1024, f);
					sscanf(buffer, "%lf %lf %lf", &trans(0, 0), &trans(0, 1), &trans(0, 2));
					fgets(buffer, 1024, f);
					sscanf(buffer, "%lf %lf %lf", &trans(1, 0), &trans(1, 1), &trans(1, 2));
					fgets(buffer, 1024, f);
					sscanf(buffer, "%lf %lf %lf", &trans(2, 0), &trans(2, 1), &trans(2, 2));

					data_.push_back(FramedTransformation2d(frame1, frame2, 0.0, trans));
				}
			}
			fclose(f);
		}
		frame_num_ = data_.size();
	}

	void SaveRelativeMatrix(std::string filename)
	{
		FILE * f = fopen(filename.c_str(), "w");
		for (int i = 0; i < (int)data_.size(); i++) {
			Eigen::Matrix3d & trans = data_[i].transformation_;
			fprintf(f, "%d\t%d %.8f\n", data_[i].frame1_, data_[i].frame2_, data_[i].score_);
			fprintf(f, "%.8f %.8f %.8f\n", trans(0, 0), trans(0, 1), trans(0, 2));
			fprintf(f, "%.8f %.8f %.8f\n", trans(1, 0), trans(1, 1), trans(1, 2));
			fprintf(f, "%.8f %.8f %.8f\n", trans(2, 0), trans(2, 1), trans(2, 2));
		}
		fclose(f);
	}

	// Only save transformation for pose.
	void SaveToFile(std::string filename, const std::vector<int>& save_id) {
		FILE * f = fopen(filename.c_str(), "w");
		for (int i = 0; i<save_id.size(); ++i)
		{
			int pose_id = save_id[i];
			Eigen::Matrix3d & trans = data_[pose_id].transformation_;
			fprintf(f, "%d\t%d\n", data_[pose_id].frame1_, data_[pose_id].frame2_);
			fprintf(f, "%.8f %.8f %.8f\n", trans(0, 0), trans(0, 1), trans(0, 2));
			fprintf(f, "%.8f %.8f %.8f\n", trans(1, 0), trans(1, 1), trans(1, 2));
			fprintf(f, "%.8f %.8f %.8f\n", trans(2, 0), trans(2, 1), trans(2, 2));
		}
		fclose(f);
	}

	void print(int id1,int id2)
	{
		std::cout << "**********************************************\n";
		std::cout << "frame " << id1 << "----" << id2 << "\n";
		for(int i=0;i<data_.size();++i)
		{
			if(data_[i].frame1_ == id1 && data_[i].frame2_ == id2)
			{
				std::cout << "score: " << data_[i].score_ << "\n";
				std::cout << data_[i].transformation_ << "\n";
			}
		}
	}
	void print(int idx)
	{
		std::cout << data_[idx].frame1_ << "\t" << data_[idx].frame2_ << "\t" << data_[idx].score_ << "\n";
		std::cout << data_[idx].transformation_ << "\n";
	}
};

// This class is used for sort raw alignments. The higher score, the more front the vertex id
struct VertexScore
{
	int vertex;
	double score;
	VertexScore():vertex(-1), score(0.0){}
	VertexScore(int v, double s):vertex(v),score(s){}
};
struct VertexScoreSort
{
	bool operator()(const VertexScore& v1, const VertexScore& v2)
	{
		return v1.score > v2.score;
	}
};
class Alignment2dSort
{
public:
	static std::vector<int> RemakeAlignment2d(const Alignments2d& alignments, Alignments2d& out_alignment)
	{
		std::vector<VertexScore> vertex_scores(alignments.frame_num_);
		for (int i=0;i<alignments.data_.size();++i)
		{
			int v1 = alignments.data_[i].frame1_;
			int v2 = alignments.data_[i].frame2_;

			vertex_scores[v1].vertex = v1;	
			vertex_scores[v1].score += alignments.data_[i].score_;
			vertex_scores[v2].vertex = v2;
			vertex_scores[v2].score += alignments.data_[i].score_;
		}
		std::sort(vertex_scores.begin(), vertex_scores.end(), VertexScoreSort());

		std::vector<int> rank_map(alignments.frame_num_);
		for (int i=0;i<vertex_scores.size();++i)
			rank_map[vertex_scores[i].vertex] = i;
		
		std::cout << "After reorganize all of raw relative transformation:\n";
		for (int i = 0; i < rank_map.size(); ++i)
			std::cout << "vertex " << i << ", rank: " << rank_map[i] << "\n";


		out_alignment.frame_num_ = alignments.frame_num_;
		for (int i=0;i<alignments.data_.size();++i)
		{
			FramedTransformation2d d;
			auto data = alignments.data_[i];
			if(rank_map[data.frame1_]<rank_map[data.frame2_])
			{
				int v1 = rank_map[data.frame1_];
				int v2 = rank_map[data.frame2_];
				d.frame1_ = v1;
				d.frame2_ = v2;
				d.transformation_ = data.transformation_;
				d.score_ = data.score_;
				out_alignment.data_.push_back(d);
			}
			else
			{
				int v1 = rank_map[data.frame1_];
				int v2 = rank_map[data.frame2_];
				d.frame1_ = v2;
				d.frame2_ = v1;
				d.transformation_ = data.transformation_.inverse();
				d.score_ = data.score_;
				out_alignment.data_.push_back(d);
			}
		}
		return rank_map;
	}

	
};





//////////////////////////////// map data structure //////////////////////////////////////////
struct IdPair {
	int frame1, frame2;
	IdPair() :frame1(0), frame2(0) {}
	IdPair(int id1, int id2) :frame1(id1), frame2(id2) {}
};

struct hash_func {
	int operator()(const IdPair& k) const
	{
		return (std::hash<int>()(k.frame1) ^ (std::hash<int>()(k.frame2) << 1) >> 1);
	}
};

struct key_equal{
	bool operator () (const IdPair &k1, const IdPair &k2) const
	{
		return k1.frame1 == k2.frame1 && k1.frame2 == k2.frame2;
	}
};

struct key_comp {
	bool operator () (const IdPair &k1, const IdPair &k2)
	{
		return k1.frame1 < k2.frame1 || (k1.frame1 == k2.frame1 && k1.frame2 < k2.frame2);
	}
};

// Give IdPair => MultiEdgeId
struct MultiEdgeId
{
	std::vector<int> edgeIds;	// multiple edges between two vertices
	int selectId;				// selected edges id between two vertices, decided by loop validate
	MultiEdgeId():selectId(-1){}
};

struct IdRank
{
	int frame1, frame2, rank;
	IdRank():frame1(0),frame2(0),rank(-1){}
	IdRank(int id1,int id2, int ranking):frame1(id1),frame2(id2),rank(ranking){}
};

struct IdRank_hash_func {
	int operator()(const IdRank& l) const
	{
		return l.frame1*1e7 + l.frame2* 1e3 + l.rank;
	}
};

struct IdRank_key_equal {
	bool operator () (const IdRank &l1, const IdRank &l2) const
	{
		return l1.frame1 == l2.frame1 && l1.frame2 == l2.frame2 && l1.rank == l2.rank;
	}
};




//////////////////////////////// Util functions //////////////////////////////////////////
class Utils
{
public:
	static std::experimental::generator<std::vector<int>> combination_generator(const int n, const int k)
	{
		std::vector<bool> v(n);
		std::fill(v.begin(), v.begin() + k, true);
		std::vector<int> comb;
		do
		{
			comb.clear();
			for (int i = 0; i<n; ++i)
				if (v[i])
					comb.push_back(i);
			yield comb;
		} while (std::prev_permutation(v.begin(), v.end()));

		comb.clear();
		yield comb;
	}

	// output permutation A_n ^k
	static std::experimental::generator<std::vector<int>> permutation_generator(const int n, const int k)
	{
		for (auto comb : combination_generator(n, k))
		{
			do
			{
				yield comb;
			} while (std::next_permutation(comb.begin(), comb.end()));
		}
		std::vector<int> perm;
		yield perm;
	}
};


//////////////////////////////// Logical empty object //////////////////////////////////////////
template<typename T>
T EmptyObject()
{
	return{};
}