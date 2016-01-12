#ifndef _OBJECTFACTORY_H
#define _OBJECTFACTORY_H

#include <vector>
#include <string>
#include <map>

// OSG:
#include <osg/Node>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/Vec3d>
#include <osg/MatrixTransform>
#include <osg/Group>
#include <osg/io_utils>
#include <osg/ShapeDrawable>
#include <osg/PositionAttitudeTransform>
#include <osg/Material>
#include <osg/StateSet>
#include <osg/LightSource>
#include <osgUtil/IntersectVisitor>
#include <osg/LineSegment>
#include <osg/PolygonMode>
#include <osg/TriangleFunctor>
#include <osg/Texture2D>
#include <osgDB/ReadFile>
#include <osgDB/FileUtils>

// Local
#include "BulletHandler.h"
#include "TriangleVisitor.h"

using namespace osg;

class ObjectFactory {
  public:
    ObjectFactory();
    virtual ~ObjectFactory();
    
    MatrixTransform* addBox( Vec3, Vec3, Quat, Vec4, bool, bool, bool );
    MatrixTransform* addSeesaw( Vec3, Vec3, Vec4, bool, bool );
    MatrixTransform* addSphere( Vec3, double, Vec4, bool, bool );
    MatrixTransform* addCylinder( Vec3, double, double, Vec4, bool, bool );
    MatrixTransform* addOpenBox( Vec3, Vec3, double, bool, bool );
    MatrixTransform* addHollowBox( Vec3, Vec3, bool, bool );
    MatrixTransform* addAntiGravityField( Vec3, Vec3, Vec3, bool );
    void addInvisibleWall( Vec3, Vec3, int );
    MatrixTransform* addPlane( Vec3, double, Vec3, bool, bool );
    PositionAttitudeTransform* addLight( Vec3, Vec4, Vec4, Vec4, StateSet* );
    MatrixTransform* addBoxHand( Vec3, Vec4 );
    MatrixTransform* addCylinderHand( double, double, Vec4 );
    MatrixTransform* addCustomObject( std::string, double, Vec3, Quat, bool, bool, float = 0.7f );
    Group* addSkybox( std::string );
    
    // Game winning
    void addGoalZone( Vec3, Vec3 );
    bool wonGame();
    void resetGame();
    
    void updateHand( Matrixd &, const Matrixd & );
    void setWorldTransform( MatrixTransform*, Matrixd & );
    void setLinearVelocity( MatrixTransform*, Vec3 );
    void updateButtonState( int );
    bool grabObject( Matrixd&, Node* );
    void releaseObject();
    
    void stepSim( double );
    BulletHandler* getBulletHandler();
    
  private:
    BulletHandler* bh;
    int numObjects;
    
    // vector of physics-connected objects
    std::vector<MatrixTransform*> m_objects;
    std::vector<float> m_scales;
    
    // vector of objects that can solve the puzzle
    // currently all spheres
    std::vector<MatrixTransform*> m_solvers;
    std::vector<int> m_physid;
    int handId;
    MatrixTransform* handMat;
    
    /* Grabbed Object Data */
    MatrixTransform* grabbedMatrix;
    Matrixd grabbedMatrixOffset;
    Drawable* grabbedShape;
    Vec4 grabbedColor;
    bool grabbedIsSD;
    int grabbedPhysId, grabbedId;
    /* End Grabbed Object Data */
    
    map<std::string, Node* > modelBank;
    
    int numLights;
    
    // goal data
    BoundingBoxd* goalBounds;
    bool m_wonGame;
};
#endif
