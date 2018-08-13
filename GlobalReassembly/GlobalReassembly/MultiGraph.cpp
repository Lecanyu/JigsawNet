#include "MultiGraph.h"


MultiGraph::MultiGraph(Alignments2d& alignments)
{
	vertexNum_ = alignments.frame_num_;

	for (int i=0;i<alignments.data_.size();++i)
	{
		auto frame1 = alignments.data_[i].frame1_;
		auto frame2 = alignments.data_[i].frame2_;
		auto score = alignments.data_[i].score_;
		auto trans = alignments.data_[i].transformation_;
		Edge edge(frame1, frame2, score, trans, i);
		edges_.push_back(edge);

		IdPair idpair(frame1, frame2);
		mapIdPair2EdgeIds_[idpair].edgeIds.push_back(i);
		mapIdPair2EdgeIds_[idpair].selectId = -1;
	}

	// assign rank
	int prev_id1 = -1, prev_id2 = -1;
	int current_rank = -1;
	for (int i = 0; i<edges_.size(); ++i)
	{
		if (edges_[i].frame1_ == prev_id1 && edges_[i].frame2_ == prev_id2)
		{
			current_rank++;
			edges_[i].rank_ = current_rank;
		}
		else
		{
			current_rank = 1;
			prev_id1 = edges_[i].frame1_;	prev_id2 = edges_[i].frame2_;
			edges_[i].rank_ = current_rank;
		}
		IdRank idrank(edges_[i].frame1_, edges_[i].frame2_, current_rank);
		mapIdRank2EdgeIds_[idrank] = i;
	}
}

MultiGraph::MultiGraph(std::string align_file, int align_file_type)
{
	Alignments2d alignments;
	// 0: line style, 1: matrix style
	if (align_file_type == 0)
		alignments.LoadFromFile(align_file);
	else if (align_file_type == 1)
		alignments.LoadFromPairwiseMartix(align_file);
	else
	{
		std::cout << "Cannot Recognize this align file type. It should 0 or 1\n";
		exit(-1);
	}
	
	std::sort(alignments.data_.begin(), alignments.data_.end(), SortFramedTransformation2d());

	vertexNum_ = alignments.frame_num_;

	for (int i = 0; i<alignments.data_.size(); ++i)
	{
		auto frame1 = alignments.data_[i].frame1_;
		auto frame2 = alignments.data_[i].frame2_;
		auto score = alignments.data_[i].score_;
		auto trans = alignments.data_[i].transformation_;
		Edge edge(frame1, frame2, score, trans, i);
		edges_.push_back(edge);

		IdPair idpair(frame1, frame2);
		mapIdPair2EdgeIds_[idpair].edgeIds.push_back(i);
		mapIdPair2EdgeIds_[idpair].selectId = -1;
	}

	// assign rank
	int prev_id1 = -1, prev_id2 = -1;
	int current_rank = -1;
	for (int i=0;i<edges_.size();++i)
	{
		if(edges_[i].frame1_ == prev_id1 && edges_[i].frame2_ == prev_id2)
		{
			current_rank++;
			edges_[i].rank_ = current_rank;
		}
		else
		{
			current_rank = 1;
			prev_id1 = edges_[i].frame1_;	prev_id2 = edges_[i].frame2_;
			edges_[i].rank_ = current_rank;
		}
		IdRank idrank(edges_[i].frame1_, edges_[i].frame2_, current_rank);
		mapIdRank2EdgeIds_[idrank] = i;
	}
}

MultiEdgeId& MultiGraph::operator()(int i, int j)
{
	IdPair idpair1(i, j);
	IdPair idpair2(j, i);
	if(mapIdPair2EdgeIds_.find(idpair1) == mapIdPair2EdgeIds_.end())
	{
		if (mapIdPair2EdgeIds_.find(idpair2) == mapIdPair2EdgeIds_.end())
			return emptyMultiEdgeObj_;		// return NULL object wrap into shared ptr.
		return mapIdPair2EdgeIds_[idpair2];
	}		
	return mapIdPair2EdgeIds_[idpair1];
}

bool MultiGraph::UnionFindSetJudgeAllLink()
{
	// init UPS data structure
	struct set_vertex
	{
		int set_id;
		std::vector<int> vertex_id;
	};
	std::vector<set_vertex> gather;			//from set to vertex
	gather.resize(vertexNum_);
	for (int i = 0; i < vertexNum_; ++i)
	{
		gather[i].set_id = i;
		gather[i].vertex_id.push_back(i);
	}
	std::vector<int> vertex2set;			//from vertex to set
	vertex2set.resize(vertexNum_);
	for (int i = 0; i < vertex2set.size(); ++i)
		vertex2set[i] = i;

	// start UPS...
	for (int i=0;i<edges_.size();++i)
	{
		if(edges_[i].select_)
		{
			int v1 = edges_[i].frame1_;
			int v2 = edges_[i].frame2_;
			//union
			if (vertex2set[v1] < vertex2set[v2])			// v2's set => v1's set
			{
				for (int t = 0; t < gather[vertex2set[v2]].vertex_id.size(); ++t)
				{
					gather[vertex2set[v1]].vertex_id.push_back(gather[vertex2set[v2]].vertex_id[t]);
				}
				for (int t = 0; t < gather[vertex2set[v2]].vertex_id.size(); ++t)
				{
					vertex2set[gather[vertex2set[v2]].vertex_id[t]] = vertex2set[v1];
				}
			}
			if (vertex2set[v1]>vertex2set[v2])
			{											// v1's set => v2's set
				for (int t = 0; t < gather[vertex2set[v1]].vertex_id.size(); ++t)
				{
					gather[vertex2set[v2]].vertex_id.push_back(gather[vertex2set[v1]].vertex_id[t]);
				}
				for (int t = 0; t < gather[vertex2set[v1]].vertex_id.size(); ++t)
				{
					vertex2set[gather[vertex2set[v1]].vertex_id[t]] = vertex2set[v2];
				}
			}
		}
	}
	

	for (int i = 1; i < vertex2set.size(); ++i)
	{
		if (vertex2set[i] != vertex2set[i - 1])
			return false;		// unconnected
	}
	return true;	// connected
}


std::experimental::generator<std::vector<int>> MultiGraph::LoopGenerator(int loop_length)
{
	int n = vertexNum_;
	int k = loop_length;
	
	if(loop_length == 3)
	{
		for (auto comb : Utils::combination_generator(n, k))
		{
			if (comb.empty()) break;
			if (LoopConnected(comb))
				yield comb;
		}
		std::vector<int> comb;
		yield comb;
	}
	else if(loop_length == 4)
	{
		for (auto comb : Utils::combination_generator(n, k))
		{
			if (comb.empty()) break;
			std::vector<int> loop(4, 0);
			for (const auto& index_perm : Parameters::isomer_4)
			{
				for (int i = 0; i<index_perm.size(); ++i)
					loop[i] = comb[index_perm[i]];
				if (LoopConnected(loop))
					yield loop;
			}
		}
		std::vector<int> comb;
		yield comb;
	}
	else if(loop_length == 6)
	{
		for (auto comb : Utils::combination_generator(n, k))
		{
			if (comb.empty()) break;
			std::vector<int> loop(6, 0);
			for (const auto& index_perm : Parameters::isomer_6)
			{
				for (int i=0;i<index_perm.size();++i)
					loop[i] = comb[index_perm[i]];
				if (LoopConnected(loop))
					yield loop;
			}
		}
		std::vector<int> comb;
		yield comb;
	}
	else
	{
		assert(false && "Unimplemented!!\n");
	}
}

std::experimental::generator<std::vector<int>> MultiGraph::LoopGeneratorGreedy(int loop_length)
{
	int n = vertexNum_;
	int k = loop_length;
	int total_size = 1;
	for (int mul=n, div = k, num=0;num<k; ++num, --mul, --div)
	{
		total_size *= mul;
		total_size /= div;
	}

	if (loop_length == 3)
	{
		std::vector<Loop> loops;
		loops.reserve(total_size);
		for (auto comb : Utils::combination_generator(n, k))
		{
			if (comb.empty()) break;
			if (LoopConnected(comb))
			{
				double score = 0.0;
				for (int i=0;i<comb.size();++i)
				{
					int v1, v2;
					if(i == comb.size()-1)
					{
						v1 = comb[i];	v2 = comb[0];
					}
					else
					{
						v1 = comb[i];	v2 = comb[i + 1];
					}
					int eid = this->operator()(v1, v2).edgeIds[0];
					score += this->operator[](eid).score_;
				}
				Loop lo(comb, score);
				loops.push_back(lo);
			}
		}
		std::sort(loops.begin(), loops.end(), LoopSort());

		for(auto& loop: loops)
		{
			yield loop.vertices;
		}
		std::vector<int> comb;
		yield comb;
	}
	else if (loop_length == 4)
	{
		std::vector<Loop> loops;
		loops.reserve(total_size * 3);
		for (auto comb : Utils::combination_generator(n, k))
		{
			if (comb.empty()) break;
			std::vector<int> loop(4, 0);
			for (const auto& index_perm : Parameters::isomer_4)
			{
				for (int i = 0; i<index_perm.size(); ++i)
					loop[i] = comb[index_perm[i]];
				if (LoopConnected(loop))
				{
					double score = 0.0;
					for (int i = 0; i<loop.size(); ++i)
					{
						int v1, v2;
						if (i == loop.size() - 1)
						{
							v1 = loop[i];	v2 = loop[0];
						}
						else
						{
							v1 = loop[i];	v2 = loop[i + 1];
						}
						
						int eid = this->operator()(v1, v2).edgeIds[0];
						score += this->operator[](eid).score_;
					}
					Loop lo(loop, score);
					loops.push_back(lo);
				}
			}
		}
		std::sort(loops.begin(), loops.end(), LoopSort());

		for (auto& loop : loops)
		{
			yield loop.vertices;
		}
		std::vector<int> comb;
		yield comb;
	}
	else
	{
		assert(false && "Unimplemented!!\n");
	}
}


bool MultiGraph::LoopConnected(const std::vector<int>& loop)
{
	for(int i=0;i<loop.size();++i)
	{
		int v1, v2;
		if (i + 1 >= loop.size())
		{
			v1 = loop[0];	v2 = loop[i];
		}
		else
		{
			v1 = loop[i];	v2 = loop[i + 1];
		}
		IdPair idpair(v1, v2);
		IdPair idpair2(v2, v1);
		if (mapIdPair2EdgeIds_.find(idpair) == mapIdPair2EdgeIds_.end() && mapIdPair2EdgeIds_.find(idpair2) == mapIdPair2EdgeIds_.end())
			return false;
	}
	return  true;
}

