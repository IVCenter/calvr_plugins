/***************************************************************
* File Name: CAVEGeometry.cpp
*
* Description:  
*
* Written by ZHANG Lelin on Nov 29, 2010
*
***************************************************************/
#include "CAVEGeometry.h"

using namespace std;
using namespace osg;


// Constructor
CAVEGeometry::CAVEGeometry(): mDOCollectorIndex(-1)
{
    mIndexClusterVector.clear();
}


// Constructor: Making deep copy of 'refGeometry' without copy of 'mDOCollectorIndex'
CAVEGeometry::CAVEGeometry(CAVEGeometry *refGeometry): mDOCollectorIndex(-1)
{
    // copy primitive set and index clusters
    unsigned int nPrimitiveSets = refGeometry->getNumPrimitiveSets();
    if (nPrimitiveSets > 0)
    {
        for (int i = 0; i < nPrimitiveSets; i++)
        {
            PrimitiveSet* primSetRef = refGeometry->getPrimitiveSet(i);

            // support primitive set 'DrawElementsUInt', add more types of primitive sets here if needed
            DrawElementsUInt* drawElementUIntRef = dynamic_cast <DrawElementsUInt*> (primSetRef);
            if (drawElementUIntRef)
            {
                unsigned int nIdices = drawElementUIntRef->getNumIndices();
                const GLenum &mode = drawElementUIntRef->getMode();

                // create duplicated primitive set, copy index field and add it to 'this'
                DrawElementsUInt* drawElementUIntDup = new DrawElementsUInt(mode, 0); 
                if (nIdices > 0)
                {
                    for (int j = 0; j < nIdices; j++) 
                    {
                        drawElementUIntDup->push_back(drawElementUIntRef->index(j));
                    }
                }
                addPrimitiveSet(drawElementUIntDup);
            }
        }
    }

    // copy the field of overlapping index by calling addIndexCluster() function sets
    const IndexClusterVector &refClusterVector = refGeometry->mIndexClusterVector;
    if (refClusterVector.size() > 0)
    {
        for (IndexClusterVector::const_iterator itrCluster = refClusterVector.begin();
            itrCluster != refClusterVector.end(); itrCluster++)
        {
            IndexClusterBase *clusterPtr = *itrCluster;
            addIndexCluster(clusterPtr);
        }
    }
}


//Destructor
CAVEGeometry::~CAVEGeometry()
{
}


/***************************************************************
*  Function: setPrimitiveSetModes()
***************************************************************/
void CAVEGeometry::setPrimitiveSetModes(const GLenum &mode)
{
    unsigned int nPrimitiveSets = getNumPrimitiveSets();
    if (nPrimitiveSets > 0)
    {
        for (int i = 0; i < nPrimitiveSets; i++)
        {
            PrimitiveSet* primset = getPrimitiveSet(i);

            // support primitive set 'DrawElementsUInt', add more types of primitive sets here if needed
            DrawElementsUInt* drawElementUIntRef = dynamic_cast <DrawElementsUInt*> (primset);
            if (drawElementUIntRef) 
                drawElementUIntRef->setMode(mode);
        }
    }
}


/***************************************************************
*  Function: addIndexCluster()
*
* 'addIndexCluster' takes record of groups of vertex indices that
*  share the same position coordinates but different normals,
*  texture coordinates and subject to different primitive sets.
*
***************************************************************/
void CAVEGeometry::addIndexCluster(const int &idx1, const int &idx2)
{
    IndexClusterBase *cluster = new IndexClusterPair(idx1, idx2);
    mIndexClusterVector.push_back(cluster);
}


void CAVEGeometry::addIndexCluster(const int &idx1, const int &idx2, const int &idx3)
{
    IndexClusterBase *cluster = new IndexClusterTriple(idx1, idx2, idx3);
    mIndexClusterVector.push_back(cluster);
}


void CAVEGeometry::addIndexCluster(const int &idx1, const int &idx2, const int &idx3, const int &idx4)
{
    IndexClusterBase *cluster = new IndexClusterQuad(idx1, idx2, idx3, idx4);
    mIndexClusterVector.push_back(cluster);
}


/***************************************************************
*  Function: addIndexCluster()
*
*  Add cluster by copying data from reference pointer 'clusterPtr'
*
***************************************************************/
void CAVEGeometry::addIndexCluster(IndexClusterBase *clusterPtr)
{
    if (clusterPtr->mNumIndices <= 1) 
        return;

    IndexClusterBase *cluster;
    const std::vector<int> &refIndexVector = clusterPtr->mIndexVector;
    if (clusterPtr->mNumIndices == 2)
    {
        cluster = new IndexClusterPair(refIndexVector[0], refIndexVector[1]);
        mIndexClusterVector.push_back(cluster);
    }
    else if (clusterPtr->mNumIndices == 3)
    {
        cluster = new IndexClusterTriple(refIndexVector[0], refIndexVector[1], refIndexVector[2]);
        mIndexClusterVector.push_back(cluster);
    }
    else if (clusterPtr->mNumIndices == 4)
    {
        cluster = new IndexClusterQuad(refIndexVector[0], refIndexVector[1], refIndexVector[2], refIndexVector[3]);
        mIndexClusterVector.push_back(cluster);
    }
}

