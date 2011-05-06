#include <GL/glew.h>
#include "GLHelper.h"
#include "RenderTraverser.h"
#include <config/ConfigManager.h>

#ifndef GL_GEOMETRY_SHADER
#define GL_GEOMETRY_SHADER 0x8DD9
#endif

RenderTraverser::RenderTraverser(std::map<int, Geometry * > * pm): mFrameID(1), mVisibilityThreshold(0),
mHierarchyRoot(NULL), mOcclusionQueries(NULL), mCurrentTestIdx(0), mIsQueryMode(false),
mNumTraversedNodes(0), mNumQueryCulledNodes(0), mNumFrustumCulledNodes(0),
mRenderTime(0), mNumRenderedGeometry(0), mUseOptimization(false), mAssumedVisibleFrames(10), pmap(pm)
{
    std::string shaderDir = cvr::ConfigManager::getEntry("Plugin.CullingMultiGPURender.ShaderDir");
    std::string cvert = shaderDir + "/philip/draw1.vert";
    std::string cgeom = shaderDir + "/philip/draw1.geom";
    GLuint _geomVert, _geomGeom;

    // initalize shaders
    createShader(cvert, GL_VERTEX_SHADER, _geomVert);
    createShader(cgeom, GL_GEOMETRY_SHADER, _geomGeom);
    createProgram(_geomProg, _geomVert, 0, _geomGeom, GL_TRIANGLES, GL_TRIANGLE_STRIP, 3);
}

RenderTraverser::RenderTraverser()
{
}


RenderTraverser::~RenderTraverser()
{
	if(mOcclusionQueries) 
		delete [] mOcclusionQueries;

        mQueryHandler.DestroyQueries();
}


void RenderTraverser::PreRender()
{
	//std::cerr << "PreRender" << std::endl;

	//std::cerr << "Frame is " << mFrameID << std::endl;

	resetMultiDrawnFlag();
	Preprocess();

	mDistanceQueue.push(mHierarchyRoot);

        RenderCoherentWithQueuePre();
	mQueryHandler.ResetQueries();
}

void RenderTraverser::PostRender()
{
	//std::cerr << "PostRender" << std::endl;
	//resetVisibleFlag();
	// Reset the draw Geometry flags

	Preprocess();
	mDistanceQueue.push(mHierarchyRoot);

        RenderCoherentWithQueuePost();

	mFrameID ++;
	mQueryHandler.ResetQueries();
}

void RenderTraverser::resetMultiDrawnFlag()
{
    // resets multidrawn flag
    std::map<int, Geometry* >::iterator it;
    for(it = pmap->begin() ; it != pmap->end(); ++it)
    {
	it->second->setDrawn(false);
	//it->second->SetFrameToCopy(frame);
    }
}

void RenderTraverser::resetVisibleFlag()
{
    std::map<int, Geometry* >::iterator it;
    for(it = pmap->begin() ; it != pmap->end(); ++it)
    {
	it->second->SetVisible(false);
    }
}

/**
	this is the algorithm as it is described in the book. It uses
	a query queue and frame-to-frame coherence in order to prevent 
	stalls and avoid unnecessary queries.
*/
void RenderTraverser::RenderCoherentWithQueuePost()
{
	//-- PART 1: process finished occlusion queries
	while(!mDistanceQueue.empty() || !queryQueue.empty())
	{
		bool resultavailable = false;

		while(!queryQueue.empty() && 
			(mDistanceQueue.empty()  || (resultavailable = queryQueue.front()->ResultAvailable())))  
		{
			while( !resultavailable && !vQueue.empty() )
			{
				HierarchyNode *node = vQueue.front();
				vQueue.pop();

				OcclusionQuery *query = IssueOcclusionQuery(node);
                                queryQueue.push(query);
                                resultavailable = queryQueue.front()->ResultAvailable();
			}

			OcclusionQuery *query = queryQueue.front();
                        queryQueue.pop();

			HandleQueryResult(query);
		}

		//-- PART 2: hierarchical traversal
		if(! mDistanceQueue.empty())
		{

			HierarchyNode *node = mDistanceQueue.top();

			mDistanceQueue.pop();
	
			mNumTraversedNodes ++;

			bool intersectsNearplane;
		
			if(InsideViewFrustum(node, intersectsNearplane))
			{

				// for near plane intersecting AABs possible 
				// wrong results => skip occlusion query
				if(intersectsNearplane)
				{
					// update node's visited flag
					node->SetLastVisited(mFrameID);
					PullUpVisibility(node->GetParent());
					node->SetVisible(true);
					/*if(node->IsLeaf())
					{
					    std::cerr << "Leaf on near plane." << std::endl;
					}*/
					TraverseNode(node);
				}
				else
				{		
					// identify previously visible nodes
					//bool wasVisible = node->Visible() && (node->LastVisited() == mFrameID - 1);
					bool wasVisible = node->Visible();
					//bool wasVisible = node->wasLastVisible(mFrameID - 1);

			
					// identify nodes that we cannot skip queries for
                                        //bool queryFeasible = (!wasVisible || (node->IsLeaf() &&
                                        //        (node->GetAssumedVisibleFrameId() <= mFrameID)));
					//bool queryFeasible = true;
					//bool queryFeasible = !wasVisible;

					// node was not recently tested => reset flag 
                                        if (node->LastVisited() != mFrameID - 1)
                                                node->SetTimesTestedInvisible(0);

					// update node's visited flag
					node->SetLastVisited(mFrameID);
			
					if(wasVisible)
					{
					    if(node->IsLeaf())
					    {
						vQueue.push(node);
					    }
					    TraverseNode(node);
					}
					else
					{
					    QueryPreviouslyInvisibleNodes(node);
					}

					/*if( queryFeasible )
					{	
						if (!wasVisible)
                                                {
							QueryPreviouslyInvisibleNodes(node);
                                                }
                                                else
                                                {
                                                        vQueue.push(node);
                                                }
					}
					else
					{
						if (node->IsLeaf())
                                                {
							node->SetVisible(true);
							PullUpVisibility(node->GetParent());
                                                }
                                                else // reset visibility classification
                                                {
                                                        node->SetVisible(false);
                                                }
					}
					
					if(wasVisible)
						TraverseNode(node);*/
				}
			}
			else
			{
				// for stats
				mNumFrustumCulledNodes ++;
			}	

		}

		if( mDistanceQueue.empty() )
		{
			IssueMultiQueries();
		}
	}

	// check for additional draws
	while (!vQueue.empty())
        {
                HierarchyNode* node = vQueue.front();
                vQueue.pop();

		OcclusionQuery *query = IssueOcclusionQuery(node);
                queryQueue.push(query);
        }

	// while there is a query available
        while (!queryQueue.empty())
        {
		OcclusionQuery *query = queryQueue.front();
                queryQueue.pop();

		// post render only draw in first context
                HandleQueryResult(query);
        }
	
}

void RenderTraverser::RenderCoherentWithQueuePre()
{
	// quick view frustrum check
	while(! mDistanceQueue.empty())
	{
		HierarchyNode *node = mDistanceQueue.top();

		mDistanceQueue.pop();
	
		bool intersectsNearplane;

		if(InsideViewFrustum(node, intersectsNearplane))
		{
		    // end traversal if leaf found (do not want to render here
		    if( !node->IsLeaf() )
			TraverseNode(node);
		}
		else  // turn off nodes that are thought to be visible
		{
		    // need to turn off all Geometry 
		    //HierarchyNode::GeometryList geometry = node->GetGeometry();
		    //for(int i = 0; i < (int) geometry.size(); i++)
		    //	geometry.at(i)->SetVisible(false);
		    node->setRecVisible(false);
		}

	}		
}

void RenderTraverser::QueryPreviouslyInvisibleNodes(HierarchyNode *node)
{
        iQueue.push(node);

        if (iQueue.size() > 50)
        {
                IssueMultiQueries();
        }
}

void RenderTraverser::IssueMultiQueries()
{
        while (!iQueue.empty())
        {
                OcclusionQuery *query = GetNextMultiQuery(iQueue);
                queryQueue.push(query);
        }
}


OcclusionQuery * RenderTraverser::GetNextMultiQuery(queue<HierarchyNode*> &iqueue)
{
        OcclusionQuery *query = mQueryHandler.RequestQuery();

        float maxBatchVal = 0.0f;
        float newPBatch = 1.0f;
        float newBatchVal;

        // issue next query
        while (!iqueue.empty())
        {
                HierarchyNode *node = iqueue.front();
                newPBatch *= (0.98f - 0.68f * exp(-(float)node->GetTimesTestedInvisible()));

                if (query->GetNodes().empty())
                {
                        // single node will anever cause a wasted query
                        newBatchVal = 1.0f;
                }
                else
                {
                        int newSize = query->GetSize() + 1;
                        newBatchVal = newSize / (1.0f + (1.0f - newPBatch) * newSize);
                }

                if (newBatchVal <= maxBatchVal)
                        break;

                iqueue.pop();
                query->AddNode(node);

                maxBatchVal = newBatchVal;
        }

        //cout <<"size: " << query->GetSize() << endl;
        IssueOcclusionQuery(*query);

        return query;
}


void RenderTraverser::HandleQueryResult(OcclusionQuery *query)
{
    //std::cerr << "Query start. size: " << query->GetSize() << std::endl;
        // wait until result available
        const int visible = query->GetQueryResult() > mVisibilityThreshold;

        // multiquery
        if (query->GetSize() > 1)
        {
                // failed query: query individual nodes
                if (visible)
                {
                        for (size_t i = 0; i < query->GetSize(); ++ i)
                        {
                                HierarchyNode *node = query->GetNodes()[i];
                                OcclusionQuery *q = IssueOcclusionQuery(node);
                                queryQueue.push(q);
                        }
                }
                else // query successful: update classifications
                {
                        for (size_t i = 0; i < query->GetSize(); ++ i)
                        {
                                HierarchyNode *node = query->GetNodes()[i];
                                node->IncTimesTestedInvisible();
                                node->setRecVisible(false);
                        }

                        mNumQueryCulledNodes += query->GetSize();
                }
        }
        else // single query
        {
                HierarchyNode *node = query->GetFrontNode();

                // failed query: query individual nodes
                if (visible)
                {
                        // node was previously invisible
                        if (!node->Visible())
                        {
                                // reset flag
                                node->SetTimesTestedInvisible(0);
                                node->SetAssumedVisibleFrameId(mFrameID + mAssumedVisibleFrames);
                        }
                        else
                        {       // randomize first invokation
                                node->SetAssumedVisibleFrameId(mFrameID + (rand() % (mAssumedVisibleFrames + 1)));
                        }

			node->SetVisible(true);
                        PullUpVisibility(node->GetParent());
                        TraverseNode(node);
                }
                else
		{
		    //std::cerr << "Rec false" << std::endl;
                        node->IncTimesTestedInvisible();
			node->setRecVisible(false);
                        ++ mNumQueryCulledNodes;
                }

                node->SetVisible(visible);
        }

	//std::cerr << "End Query result." << std::endl;
}




void RenderTraverser::TraverseNode(HierarchyNode *node)
{
	
	mNumTraversedNodes ++;

	if(node->IsLeaf())
	{
		//std::cerr << "Render called on Non-MultiDrawn node." << std::endl;
		glUseProgram(_geomProg);
		mNumRenderedGeometry += node->Render();
		glUseProgram(0);
	}
	else // internal node: add children to priority queue for further processing
	{
		mDistanceQueue.push(node->GetLeftChild());
		mDistanceQueue.push(node->GetRightChild());
	}
}


void RenderTraverser::RenderVisualization()
{
	mDistanceQueue.push(mHierarchyRoot);

	while(!	mDistanceQueue.empty())
	{
		HierarchyNode *node = mDistanceQueue.top();
		mDistanceQueue.pop();

		// identify previously visible nodes
		bool wasVisible = node->Visible() && (node->LastVisited() == mFrameID - 1);

		if(wasVisible)
			TraverseNode(node);
		else
		{
			// also render culled nodes
			glColor3f(1.0,0.0,0.0);
			node->RenderBoundingVolumeForVisualization();		
		}
	}
}


void RenderTraverser::PullUpVisibility(HierarchyNode *node)
{
	while(node && !node->Visible())
	{
		node->SetVisible(true);
		node = node->GetParent();
	}
}

bool RenderTraverser::ResultAvailable(HierarchyNode *node)
{
	int result;

	glGetQueryivARB(node->GetOcclusionQuery(),
					GL_QUERY_RESULT_AVAILABLE_ARB, &result);

	//printf("Occulsion result available %d\n", result);
	return (result == GL_TRUE);
}

void RenderTraverser::SetHierarchy(HierarchyNode *sceneRoot)
{
    //std::cerr << "Hierarchy change." << std::endl;
	while(! vQueue.empty() )
	    vQueue.pop();
	mHierarchyRoot = sceneRoot;
	mHierarchyRoot->findNodeVisibility();
	mHierarchyRoot->setAllLastVis(mFrameID - 1);

}

HierarchyNode *RenderTraverser::GetHierarchy()
{
	return mHierarchyRoot;
}

int RenderTraverser::GetOcclusionQueryResult(HierarchyNode *node)
{
	unsigned int result;
	
	glGetQueryObjectuivARB(node->GetOcclusionQuery(), GL_QUERY_RESULT_ARB, &result);

	return (int)result;
}


void RenderTraverser::Switch2GLQueryState()
{	
	// boolean used to avoid unnecessary state changes
	if(!mIsQueryMode)
	{
		glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
		glDepthMask(GL_FALSE);
		glDisable(GL_LIGHTING);
		mIsQueryMode = true;
	}
}


void RenderTraverser::Switch2GLRenderState()
{
	// boolean used to avoid unnecessary state changes
	if(mIsQueryMode)
	{
		// switch back to rendermode		
		glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
		glDepthMask(GL_TRUE);
		glEnable(GL_LIGHTING);
		mIsQueryMode = false;
	}
}

OcclusionQuery *RenderTraverser::IssueOcclusionQuery(HierarchyNode *node)
{
        OcclusionQuery *query = mQueryHandler.RequestQuery();
        query->AddNode(node);

        IssueOcclusionQuery(*query);

        return query;
}


void RenderTraverser::IssueOcclusionQuery(const OcclusionQuery &query)
{
        ++ mNumIssuedQueries;

        query.BeginQuery();

	Switch2GLQueryState();
	vector<HierarchyNode* > nodes = query.GetNodes();
	for(int i = 0; i < (int) nodes.size(); i++)
		nodes.at(i)->RenderBoundingVolume();
	Switch2GLRenderState();

        query.EndQuery();
}


void RenderTraverser::IssueOcclusionQuery(HierarchyNode *node, bool wasVisible)
{
	// get next available test id
	unsigned int occlusionQuery = mOcclusionQueries[mCurrentTestIdx++];
	
	node->SetOcclusionQuery(occlusionQuery);
	// do the actual occlusion query for this node
	glBeginQueryARB(GL_SAMPLES_PASSED_ARB, occlusionQuery);
	
	// if leaf and was visible => will be rendered anyway, thus we
	// can also test with the real geometry 
	if(node->IsLeaf() && wasVisible && mUseOptimization)
	{
		mNumRenderedGeometry += node->Render();
	}
	else
	{
		// change state so the bounding box gets not actually rendered on the screen
		Switch2GLQueryState();
		node->RenderBoundingVolume();
		Switch2GLRenderState();
	}

	glEndQueryARB(GL_SAMPLES_PASSED_ARB);
}

void RenderTraverser::Preprocess()
{
	// view frustum planes for view frustum culling
	calcViewFrustumPlanes(&mClipPlanes, mProjViewMatrix);
	calcAABNPVertexIndices(mNPVertexIndices, mClipPlanes);
	// generate ids for occlusion test
	
	mCurrentTestIdx = 0;

	// reset statistics
	mNumTraversedNodes = 0;
	mNumQueryCulledNodes = 0;
	mNumFrustumCulledNodes = 0;
	mNumRenderedGeometry = 0;
}


void RenderTraverser::SetViewpoint(Vector3 const &viewpoint)
{
	copyVector3(mViewpoint, viewpoint);
}
	

void RenderTraverser::SetProjViewMatrix(Matrix4x4 const &projViewMatrix)
{
	copyMatrix(mProjViewMatrix, projViewMatrix);
}


bool RenderTraverser::InsideViewFrustum(HierarchyNode *node, bool &intersectsNearplane)
{
	Vector3x8 vertices;
	
	calcAABoxPoints(vertices, node->GetBoundingVolume());

	// test all 6 clip planes if a bouning box vertex is outside
	// only need the n and p vertices of the bouding box to determine this
	for (int i = 0; i < 6; i++)
	{		
		// test the n-vertex
		// note: the calcAABNearestVertexId should be preprocessed
		if(!pointBeforePlane(mClipPlanes.plane[i], vertices[mNPVertexIndices[i * 2]]))
		{
			// outside
			return false;
		}
	}

	// test if bounding box is intersected by nearplane (using the p-vertex)
	intersectsNearplane = (!pointBeforePlane(mClipPlanes.plane[5], vertices[mNPVertexIndices[11]]));

	// -- get vector from viewpoint to center of bounding volume
	Vector3 vec;
	calcAABoxCenter(vec, node->GetBoundingVolume());
	diffVector3(vec, vec, mViewpoint);

	// compute distance from nearest point to viewpoint
	diffVector3(vec, vertices[calcAABNearestVertexIdx(vec)], mViewpoint);
	node->SetDistance(squaredLength(vec));

	return true;
}


void RenderTraverser::SetVisibilityThreshold(int threshold)
{
	mVisibilityThreshold = threshold;
}

long RenderTraverser::GetRenderTime()
{
	return mRenderTime;
}

int RenderTraverser::GetNumTraversedNodes()
{
	return mNumTraversedNodes;
}

int RenderTraverser::GetNumQueryCulledNodes()
{
	return mNumQueryCulledNodes;
}

int RenderTraverser::GetNumFrustumCulledNodes()
{
	return mNumFrustumCulledNodes;
}


int RenderTraverser::GetNumRenderedGeometry()
{
	return mNumRenderedGeometry;
}


int RenderTraverser::GetVisibilityThreshold()
{
	return mVisibilityThreshold;
}

void RenderTraverser::SetUseOptimization(bool useOptimization)
{
	mUseOptimization = useOptimization;
}
