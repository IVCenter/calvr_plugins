/***************************************************************
* File Name: CAVEGeodeIconSurface.h
*
* Description: Derived class from CAVEGeodeIcon
*
***************************************************************/

#ifndef _CAVE_GEODE_ICON_SURFACE_H_
#define _CAVE_GEODE_ICON_SURFACE_H_


// C++
#include <iostream>
#include <list>

// Open scene graph
#include <osg/BoundingBox>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/StateSet>

// local
#include "CAVEGeodeIcon.h"


/***************************************************************
* Class: CAVEGeodeIconSurface
***************************************************************/
class CAVEGeodeIconSurface: public CAVEGeodeIcon
{
  public:
    CAVEGeodeIconSurface(osg::Vec3Array **surfVertexArray, osg::Vec3Array **surfNormalArray,
			 osg::Vec2Array **surfTexcoordArray, CAVEGeometry **surfGeometry, CAVEGeometry **surfGeometryRef);
    ~CAVEGeodeIconSurface();

    virtual void movedon() {}
    virtual void movedoff() {}
    virtual void pressed() {}
    virtual void released() {}

    CAVEGeometry *getGeometry() { return mGeometry; }
    CAVEGeometry *getGeometryOriginPtr() { return mGeometryOriginPtr; }
    CAVEGeometry *getGeometryReferencePtr() { return mGeometryReferencePtr; }

    /* icon surface geode highlighting functions */
    void setHighlightNormal();
    void setHighlightSelected();
    void setHighlightUnselected();

  protected:

    CAVEGeometry *mGeometry;			// geometry that used to render 'CAVEGeodeIconSurface'
    CAVEGeometry *mGeometryOriginPtr;		// geometry pointer to track the origin 'CAVEGeometry' that to be editted
    CAVEGeometry *mGeometryReferencePtr;	// geometry pointer to track the ghost 'CAVEGeometry'

    /* static members that decides transparency of different color & texture states
      'gAlphaNormal': normal alpha value when 'CAVEGeodeShape' is picked by 'DOGeometryCollector'
      'gAlphaSelected': alpha value when a specific 'CAVEGeodeIconSurface' is selected
      'gAlphaUnselected': alpha value when a specific 'CAVEGeodeIconSurface' is not selected
    */
    static const float gAlphaNormal;
    static const float gAlphaSelected;
    static const float gAlphaUnselected;

    static const osg::Vec3 gDiffuseColorNormal;
    static const osg::Vec3 gDiffuseColorSelected;
    static const osg::Vec3 gDiffuseColorUnselected;
    static const osg::Vec3 gSpecularColor;
};








#endif
