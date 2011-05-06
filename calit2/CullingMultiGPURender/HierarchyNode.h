#ifndef HIERARCHYNODE_H
#define HIERARCHYNODE_H

#include "Geometry.h"

extern "C" {
   #include "DataTypes.h"
}

#include <vector>
#include <stack>
#include <iostream>
#include <fstream>


using namespace std;

class HierarchyNode;

/** @file CallbackDrawable.h	
 * This class implements the compare operator for the priority queue.
 * a lower distance has a higher value in the queue
*/
template <typename T> class myless
{
public:
	
	//bool operator() (HierarchyNode *v1, HierarchyNode *v2) const
	bool operator() (T v1, T v2) const
    {
		return (v1->mDistance > v2->mDistance);
    }
};

/** 
	This class represents a node in a k-d tree hierarchy. A
	node has two children. The node can be either an interior 
	node (i.e., left and right child NULL) or a leaf node, which holds 
	the actual geometry.
*/

class HierarchyNode
{
	template <typename T> friend class myless;

public:
	typedef stack<HierarchyNode *> TraversalStack;
    	typedef vector<Geometry *> GeometryList;

	HierarchyNode();
	//! generates hierarchynode with specified upper and lower bounding volume points on specified depth
	HierarchyNode(const Vector3 boundLower, const Vector3 boundUpper, 
				 HierarchyNode *parent, int depth);

	~HierarchyNode();
	//! was this node visible the last time it was visited?
	bool Visible();
	//! last time this node was visited (in framenumber)
	int LastVisited();
	//! sets visible flag
	void SetVisible(bool visible);
	//! sets timestamp (current framenumber)
	void SetLastVisited(int lastVisited);
	//! is this node a leaf node (i.e., geometry)
	bool IsLeaf();
	//! renders the geometry in this node. returns number of rendered geometry
	int Render();
	//! returns occlusion query id
	int GetOcclusionQuery();
	//! sets occlusion query id
	void SetOcclusionQuery(int occlusionQuery);
	//! renders the bounding volume  (i.e., a axis aligned bounding box) of the node
	void RenderBoundingVolume();
	//! adds geometry to this node (rendered if this node is a leaf)
	void AddGeometry(Geometry *geometry);

        bool findNodeVisibility();
        void setAllLastVis(int frame);
        void setRecVisible(bool b);

        int getGeometryLastRendered();

	void IncTimesTestedInvisible();
	int GetTimesTestedInvisible() const;
	void SetTimesTestedInvisible(int t);
	void SetAssumedVisibleFrameId(int t);
	int GetAssumedVisibleFrameId() const;
	int GetNumLeafs();

	void SetLeftChild(HierarchyNode *child);
	void SetRightChild(HierarchyNode *child);

	HierarchyNode *GetParent();
    	HierarchyNode *GetLeftChild();
	HierarchyNode *GetRightChild();

	//! compute bounding volume (i.e., a axis aligned bounding box) for this geometry.
	//void CalcBoundingVolume();
	const AABox& GetBoundingVolume();
	const AABox& GetEnclosedSpace();
	
	//! generates the kd-tree from this root-node, returns number of nodes in hierarchy.
	int GenerateKdTree();

	//! returns number of nodes in hierarchy with this node.
	int GetNumHierarchyNodes();

	//! add children of this node ordered to a traversal stack with respect to the view point
	void PushChildrenOrdered(const Vector3 viewpoint, TraversalStack &traversalStack);

	// set frame
	void SetFrame(int fram) { frame = fram; };

	//! returns geometry list
	GeometryList &GetGeometry();

	//! visibly renders also bounding volume of this node
	static void SetRenderBoundingVolume(bool renderBoundingVolume);
        void RenderBoundingVolumeForVisualization();
        	
	/**
		initialises the k-d tree and the static termination criteria.
		this method must be called after all geometry has been added to the root node
		and before the tree has been generated.
	*/
	static void InitKdTree(HierarchyNode *root);

	float GetDistance();
	void SetDistance(float distance);

        bool getMultiDrawn() { return _multiDrawn; }
        void setMultiDrawn(bool b) { _multiDrawn = b; }

	float mDistance;

        bool wasLastVisible(int frame);
        void setLastVisible(int frame);

protected:

	//! generates hierarchynode with specified upper and lower bounding volume points on specified depth
	//HierarchyNode(const Vector3 boundLower, const Vector3 boundUpper, 
	//			 HierarchyNode *parent, int depth);

	enum {X_AXIS, Y_AXIS, Z_AXIS};

	//! criteria to stop the splitting of this kd-tree node
	bool SimpleEnough();
	//! split plane of the kd-tree (according to mSplitAxis, it is the x, y, or z axis)
	float ComputeSplitPlane();

	/** calculates a value that expresses if it's reasonable to split the 
	   node at the specified position (the smaller the value the better the split)
	   */
	float ComputeHeuristics(float pos);
	
	// node specific geometry
	GeometryList mGeometry;
	//! id of this occlusion query of this node
	int mOcclusionQuery;

	HierarchyNode *mLeftChild;	
	HierarchyNode *mRightChild;
	HierarchyNode *mParent;
	
	bool  mVisible;
	int mLastVisited;
	bool mAABValid;
	bool mEnclosedSpaceValid;
	int mNumHierarchyNodes;
	int mTimesInvisible;
	int mAssumedVisibleFrameId;
	
	AABox mBoundingBox;
	// the enclosed space of the pure kd-treenode (without considering the geometry)
	AABox mEnclosedSpace;
	
	int mSplitAxis;
	int mLastRendered;
	float mSplitValue;
	int mDepth;
	int frame;

	static bool sRenderBoundingVolume;
	static float sSplitBandwith;

	// --termination criteria

	static int sGeometryThreshold;
	// the maximum bounding box surface
	static float sSurfaceThreshold;
	// the maximum tree depth
	static int sMaxDepth;
	static int sNumLeafs;
	// the default bounding box drawing color
	float mBoxColor[3];
	// distance to the view point
        //

        bool _multiDrawn;
};

#endif // HIERARCHYNODE_H
