#ifndef _OcclusionQuery_H__
#define _OcclusionQuery_H__

#include <vector>
#include "HierarchyNode.h"



class HierarchyNode;

/** @file OcclusionQuery.h
 *  This class is an implementation for single node queries and multiqueries.	
 *  @remark the class encapsulates hardware occlusion query calls.
*/
class OcclusionQuery
{

        friend class QueryHandler;

public:
	/** constructor requesting an opengl occlusion query.
	*/
	OcclusionQuery();

	virtual ~OcclusionQuery();

	bool ResultAvailable() const;
	
	unsigned int GetQueryResult() const;
	
	void BeginQuery() const;
	
	void EndQuery() const;
	
	unsigned int GetQueryId() const;
	/** Returns the first node of the multiquery
	*/
	inline HierarchyNode *GetFrontNode() const { return mNodes[0]; }
	inline const vector<HierarchyNode*> &GetNodes() const { return mNodes; }

	/** Reset the list of nodes associated with this query.
	*/
	inline void Reset() { mNodes.clear(); }
	/** Adds a node to the query.
	*/
	inline void AddNode(HierarchyNode *node) { mNodes.push_back(node); }
	/** Returns the size of the multiquery.
	*/
	inline int GetSize() const { return (int)mNodes.size();}
	
protected:

	///////
	//-- members
        OcclusionQuery(unsigned int id): mQueryId(id) { }	

	/// all nodes that are tested with the same query
	vector<HierarchyNode*> mNodes; 
	// the query associated with this test
	unsigned int mQueryId;
};


class QueryHandler
{
public:

	QueryHandler();
	~QueryHandler() { DestroyQueries(); }

	OcclusionQuery *RequestQuery();

	/** Must be called every frame.
	*/
	void ResetQueries();
	/** Destroys all the queries.
	*/
	void DestroyQueries();

protected:
	
	/** allocates n queries in advance
	*/
	void Allocate(int n);


	////////////////

	int mCurrentQueryIdx;

	vector<OcclusionQuery*> mOcclusionQueries;
};
#endif // OcclusionQuery_H
