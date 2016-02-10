#ifndef _BULLETHANDLER_H
#define _BULLETHANDLER_H
#define BIT(x) (1<<(x))

#include <vector>
#include <map>
#include <string>
#include <osg/Vec3>
#include <osg/Geometry>
#include <osg/Matrixd>
#include <osg/io_utils>
#include <btBulletDynamicsCommon.h>
#include <btBulletCollisionCommon.h>
#include "AntiGravityField.h"
#include "TriangleVisitor.h"
#include <BulletCollision/Gimpact/btGImpactShape.h>

enum CollisionType {
  COL_NORMAL = BIT(1),
  COL_SPHERE = BIT(2),
  COL_WALL = BIT(3)
};

class BulletHandler
{
  public:
    BulletHandler();
    virtual ~BulletHandler();
    
    int addBox( osg::Vec3, osg::Vec3, osg::Quat, bool );
    int addSeesaw( osg::Vec3, osg::Vec3, bool );
    int addSphere( osg::Vec3, double, bool );
    int addCylinder( osg::Vec3, osg::Vec3, bool );
    int addOpenBox( osg::Vec3, osg::Vec3, double, bool );
    int addHollowBox( osg::Vec3, osg::Vec3, bool );
    int addCustomObject( std::string, std::vector<Triangle> *, double, osg::Vec3, osg::Quat, bool, float = 0.7f );
    void addAntiGravityField(osg::Vec3, osg::Vec3, osg::Vec3);
    void addInvisibleWall(osg::Vec3, osg::Vec3, int);
    
    void setLinearVelocity( int, osg::Vec3 );
    osg::Vec3 getLinearVelocity( int );
    void activate( int );
    void translate( int, osg::Vec3 );
    void setGravity( osg::Vec3 );
    
    void stepSim( double );
    btDiscreteDynamicsWorld* getDynamicsWorld();
    void getWorldTransform( int, osg::Matrixd& );
    void setWorldTransform( int, osg::Matrixd& );
    
    void addHand(osg::Vec3, osg::Vec3);
    void moveHand( osg::Matrixd& );
    void updateButtonState( int );
    
  private:
    btBroadphaseInterface* broadphase;
    btDefaultCollisionConfiguration* btcc;
    btCollisionDispatcher* btcd;
    btSequentialImpulseConstraintSolver* btsolver;
    btDiscreteDynamicsWorld* dynamicsWorld;
    
    btRigidBody* addRigid( btCollisionShape*, btDefaultMotionState*, CollisionType, CollisionType, bool, float = 0.7f );
    std::vector<btRigidBody*> rbodies;
    int numRigidBodies;
    
    // DOES NOT WORK BECAUSE THE OBJECT CAN BE A DIFFERENT SCALE (needs to be rethought)
    //std::map<std::string,btCollisionShape*> meshBank;
    
    std::vector<AntiGravityField*> avfs;
    int numavfs;
};
#endif
