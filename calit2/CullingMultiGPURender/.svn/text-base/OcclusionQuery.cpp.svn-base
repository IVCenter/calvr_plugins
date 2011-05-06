#include <GL/glew.h>
#include "OcclusionQuery.h"
#include <iostream>

using namespace std;


OcclusionQuery::OcclusionQuery()
{
	glGenQueriesARB(1, &mQueryId);

	// reverse for multiqueries with 20 nodes
	mNodes.reserve(32);
}


OcclusionQuery::~OcclusionQuery()
{
	glDeleteQueriesARB(1, &mQueryId);
}


void OcclusionQuery::BeginQuery() const
{
	glBeginQueryARB(GL_SAMPLES_PASSED_ARB, mQueryId);
}


void OcclusionQuery::EndQuery() const
{
	glEndQueryARB(GL_SAMPLES_PASSED_ARB);
}


unsigned int OcclusionQuery::GetQueryId() const
{
	return mQueryId;
}


bool OcclusionQuery::ResultAvailable() const
{
	GLuint available = GL_FALSE;

	glGetQueryObjectuivARB(mQueryId, GL_QUERY_RESULT_AVAILABLE_ARB, &available);

	return available == GL_TRUE;
}


unsigned int OcclusionQuery::GetQueryResult() const
{
	GLuint sampleCount = 1;

	glGetQueryObjectuivARB(mQueryId, GL_QUERY_RESULT_ARB, &sampleCount);
	
	return sampleCount;
}



QueryHandler::QueryHandler(): mCurrentQueryIdx(0) 
{
	Allocate(10000);
}


OcclusionQuery *QueryHandler::RequestQuery()
{
	OcclusionQuery *query;

	if (mCurrentQueryIdx == (int) mOcclusionQueries.size())
	{
		query = new OcclusionQuery();
		mOcclusionQueries.push_back(query);
	}
	else
	{
		query = mOcclusionQueries[mCurrentQueryIdx];
		query->Reset();
	}

	++ mCurrentQueryIdx;

	return query;
}


void QueryHandler::ResetQueries()
{
	mCurrentQueryIdx = 0;
}


void QueryHandler::DestroyQueries()
{
	while((int)mOcclusionQueries.size())
	{
		delete mOcclusionQueries.at(0);
		mOcclusionQueries.pop_back();
	}
	mCurrentQueryIdx = 0;
	mOcclusionQueries.clear();
}


void QueryHandler::Allocate(int n)
{
	unsigned int *ids = new unsigned int[n]; 
	glGenQueriesARB(n, ids);

	for (int i = 0; i < n; ++ i) 
	{
		OcclusionQuery *q = new OcclusionQuery(ids[i]);
		mOcclusionQueries.push_back(q);
	}

	mCurrentQueryIdx = n;
	delete [] ids;
}
