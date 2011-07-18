#ifndef _CYLINDER_DRAWABLE_H_
#define _CYLINDER_DRAWABLE_H_

#include <osg/Geometry>
#include <osg/Vec3>

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <GL/glut.h>
#include <iostream>
#include <vector>

#include "PanoDrawable.h"

#include <OpenThreads/Mutex>
#include <osgDB/ReadFile>

class CylinderDrawable : public PanoDrawable
{
  public:
    CylinderDrawable(float radius_in, float viewangle_in,float viewangleh_in, float camHeight_in, int segmentsPerTexture_in, int maxTextureSize_in);
    void drawImplementation(osg::RenderInfo&) const;

    void setFlip(int f);

    void setImage(std::string file_path);
    void setImage(std::string file_path_r, std::string file_path_l);

    void updateRotate(float);

    float getRadius();
    void setRadius(float r);
    int getSegmentsPerTexture();
    void setSegmentsPerTexture(int spt);
    int getMaxTextureSize();
    void setMaxTextureSize(int mts);
    void getViewAngle(float & a, float & b);
    void setViewAngle(float a, float b);
    float getCamHeight();
    void setCamHeight(float h);

    void deleteTextures();
    bool deleteDone();

    /** Copy constructor using CopyOp to manage deep vs shallow copy.*/
    CylinderDrawable(const CylinderDrawable&,const osg::CopyOp& copyop=osg::CopyOp::SHALLOW_COPY);
    virtual Object* cloneType() const { return new CylinderDrawable(30000, 120, 360.0, 0.0, 100, 4096); }
    virtual Object* clone(const osg::CopyOp& copyop) const { return new CylinderDrawable(*this,copyop); }
    virtual bool isSameKindAs(const Object* obj) const { return dynamic_cast<const CylinderDrawable*>(obj)!=NULL; }
    virtual const char* libraryName() const { return "Cylinder"; }
    virtual const char* className() const { return "CylinderDrawable"; }

  protected:

    enum eye
    {
        RIGHT = 1,
        LEFT = 2,
        BOTH = 3
    };

    mutable int currenteye;

    mutable int badinit;

    bool initTexture(eye e, int context) const;

    float _rotation;

    static OpenThreads::Mutex _initLock;

    mutable bool _deleteDone, _doDelete;
    mutable int rows, cols; 
    float radius;
    float viewangle;
    float viewangleh;
    float camHeight, floorOffset;
    mutable std::string rfile, lfile;
    //GLuint * textures;
    mutable int segmentsPerTexture, maxTextureSize, width, height, init, mono, flip;
    virtual ~CylinderDrawable();

    mutable std::vector<std::vector< unsigned char * > > rtiles;
    mutable std::vector<std::vector< unsigned char * > > ltiles;
    static std::map<int, std::vector<std::vector< GLuint * > > > rtextures;
    static std::map<int, std::vector<std::vector< GLuint * > > > ltextures;
    static std::map<int, int> _contextinit;
    mutable int _maxContext;

    bool _renderOnMaster;
};
#endif
