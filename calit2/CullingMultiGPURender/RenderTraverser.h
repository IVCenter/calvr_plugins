#ifndef RENDERTRAVERSER_H
#define RENDERTRAVERSER_H

#include "Geometry.h"
#include "FetchQueue.h"
#include "HierarchyNode.h"
#include "OcclusionQuery.h"
#include "math.h"
#include <queue>
#include <stack>
#include <map>

using namespace std;

typedef stack<HierarchyNode *> TraversalStack;
typedef queue<OcclusionQuery *> QueryQueue;
typedef priority_queue<HierarchyNode *, vector<HierarchyNode *>, myless<vector<HierarchyNode *>::value_type> > PriorityQueue;

/** @file RenderTraversal.h
 *  RenderTraversal class deals with traversal of the tree, first eliminating geometry
 *  via view frustrum and then applying occulsion culling in a iterative process.
 */
class RenderTraverser
{
public:
	RenderTraverser(std::map<int , Geometry* > * pm);
	~RenderTraverser();
	void PreRender();
	void PostRender();
	//! sets the scene hierarchy.
	void SetHierarchy(HierarchyNode *sceneRoot);
	//! sets viewpoint
	void SetViewpoint(Vector3 const &viewpoint);
	//! sets view projection matrix
	void SetProjViewMatrix(Matrix4x4 const &projViewMatrix);
	//! returns root of hierarchy
	HierarchyNode *GetHierarchy();
	//! sets visible pixels threshold for visibility classification
	void SetVisibilityThreshold(int threshold);
	//! returns visibility threshold
	int GetVisibilityThreshold();

	//! returns rendering time of the specified algorihtm
	long GetRenderTime();
	//! returns number of traversed nodes
	int  GetNumTraversedNodes();
	//! returns the number of hierarchy nodes culled by the occlusion query
	int GetNumQueryCulledNodes();
	//! returns the number of hierarchy nodes culled by the frustum culling only
	int GetNumFrustumCulledNodes();
	//! returns number of rendered geometric objects (e.g., teapots, ...)
	int GetNumRenderedGeometry();

	//! renders a visualization of the hierarchy
	void RenderVisualization();

	//! use optimization to take leaf nodes instead of bounding box for occlusion queries	
	void SetUseOptimization(bool useOptimization);

	OcclusionQuery* IssueOcclusionQuery(HierarchyNode *node);
	void IssueOcclusionQuery(const OcclusionQuery &query);
	void HandleQueryResult(OcclusionQuery *query);
	void QueryPreviouslyInvisibleNodes(HierarchyNode *node);
	void IssueMultiQueries();
	OcclusionQuery * GetNextMultiQuery(queue<HierarchyNode*> &iqueue);

	void SetFrame(int fram) {frame = fram;};


        void resetMultiDrawnFlag();
        void resetVisibleFlag();

protected:
        RenderTraverser();

	//! renders the scene with the coherent hierarchical algorithm and the query queye
	void RenderCoherentWithQueuePre();
	void RenderCoherentWithQueuePost();
	//! does some importand initialisations
	void Preprocess();
	
	//! returns occlusion query result for specified node
	int GetOcclusionQueryResult(HierarchyNode *node);
	//! the node is traversed as usual
	void TraverseNode(HierarchyNode *node);
	//! visibility is pulled up from visibility of children 
	void PullUpVisibility(HierarchyNode *node);
	//! is result available from query queue?
	bool ResultAvailable(HierarchyNode *node);
	//! issues occlusion query for specified node
	void IssueOcclusionQuery(HierarchyNode *node, bool wasVisible);
	void IssueOcclusionQuery(vector<HierarchyNode*> nodes);
	//! true if bounding box is culled by view frustum culling
	/**
		intersectsNearplane returns true if bounding box intersects the near plane.
		additionally stores the distance from the near plane to the center of the 
		current node with the node. this will for front-to-back ordering
	*/
	bool InsideViewFrustum(HierarchyNode *node, bool &intersects);
	//! switches to normal render mode
	void Switch2GLRenderState();
	//! switches to occlusion query mode (geometry not rendered on the screen)
	void Switch2GLQueryState();

protected:

	// the current clip planes of the view frustum
	VecPlane mClipPlanes;
	// the indices of the np-vertices of the bounding box for view frustum culling
	int mNPVertexIndices[12];

	Vector3 mViewpoint;
	Matrix4x4 mProjViewMatrix;
	HierarchyNode *mHierarchyRoot;
        GLuint _geomProg;
	
	//! we use a priority queue rather than a renderstack
	PriorityQueue mDistanceQueue; 
	QueryQueue queryQueue;	
	queue<HierarchyNode*> iQueue;
	queue<HierarchyNode*> vQueue;
	QueryHandler mQueryHandler;
	
	int mFrameID;
	int mVisibilityThreshold;
	unsigned int *mOcclusionQueries;
	int mCurrentTestIdx;
	bool mIsQueryMode;
        std::map<int, Geometry* > *pmap;

        bool drawFirstContext;
		
	// statistics
	int mNumTraversedNodes;
	int mNumQueryCulledNodes;
	int mNumFrustumCulledNodes;
	int mNumRenderedGeometry;
	int mAssumedVisibleFrames;
	int mNumIssuedQueries;
	int mPreFetched;
	int mDataCopied;
	int mDataTransfered;

	long mRenderTime;

	bool mUseOptimization;

	int frame;
};

#endif // RENDERTRAVERSER_H
