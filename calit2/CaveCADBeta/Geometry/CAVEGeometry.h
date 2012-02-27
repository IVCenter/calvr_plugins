/***************************************************************
* File Name: CAVEGeometry.h
*
* Class Name: CAVEGeometry
*
***************************************************************/

#ifndef _CAVE_GEOMETRY_H_
#define _CAVE_GEOMETRY_H_


// C++
#include <iostream>
#include <list>
#include <vector>

// Open scene graph
#include <osg/Geometry>


class CAVEGeometry;
typedef std::vector<CAVEGeometry*>	CAVEGeometryVector;


/***************************************************************
* Class: CAVEGeometry
***************************************************************/
class CAVEGeometry: public osg::Geometry
{
    /* allow class 'DOGeometryCollector' and 'CAVEGeodeShape' to change its private vector index values */
    friend class DOGeometryCollector;
    friend class CAVEGeodeShape;

  public:
    CAVEGeometry();
    CAVEGeometry(CAVEGeometry *refGeometry);
    ~CAVEGeometry();

    /* This function is called by 'CAVEGeodeEditWireframe' that converts all polygons and triangles to line strips */
    void setPrimitiveSetModes(const GLenum &mode);

    /***************************************************************
    * Class: IndexClusterBase
    ***************************************************************/
    class IndexClusterBase
    {
      public:
	IndexClusterBase(): mNumIndices(0) {}

	/* number of indices contained in a cluster */
	int mNumIndices;
	std::vector<int> mIndexVector;
    };

    typedef std::vector<IndexClusterBase*>	IndexClusterVector;

    /***************************************************************
    * Derived classes from 'IndexClusterBase', data structures that
    * hold index clusters that shared the same coordinates, a normal
    * cluster size is limited to four index integers
    ***************************************************************/
    class IndexClusterPair: public IndexClusterBase
    {
      public:
	IndexClusterPair(const int &idx1, const int &idx2)
	{
	    mNumIndices = 2;
	    mIndexVector.push_back(idx1);
	    mIndexVector.push_back(idx2);
	}
    };

    class IndexClusterTriple: public IndexClusterBase
    {
      public:
	IndexClusterTriple(const int &idx1, const int &idx2, const int &idx3)
	{
	    mNumIndices = 3;
	    mIndexVector.push_back(idx1);
	    mIndexVector.push_back(idx2);
	    mIndexVector.push_back(idx3);
	}
    };

    class IndexClusterQuad: public IndexClusterBase
    {
      public:
	IndexClusterQuad(const int &idx1, const int &idx2, const int &idx3, const int &idx4)
	{
	    mNumIndices = 4;
	    mIndexVector.push_back(idx1);
	    mIndexVector.push_back(idx2);
	    mIndexVector.push_back(idx3);
	    mIndexVector.push_back(idx4);
	}
    };

    /***************************************************************
    * 'addIndexCluster' takes record of groups of vertex indices
    *  sharing the same position coordinates but different normals,
    *  texture coordinates and subject to different primitive sets.
    *
    ***************************************************************/
    void addIndexCluster(const int &idx1, const int &idx2);
    void addIndexCluster(const int &idx1, const int &idx2, const int &idx3);
    void addIndexCluster(const int &idx1, const int &idx2, const int &idx3, const int &idx4);

    void addIndexCluster(IndexClusterBase *clusterPtr);

  protected:

    /* index cluster vector that takes record of all vertex clusters that sharing the same coordinates */
    IndexClusterVector mIndexClusterVector;

    /* vector index that indicates the selection status of the Geometry, ONLY accessed by 'DOGeometryCollector' */
    int mDOCollectorIndex;
};


#endif
