#include <sys/time.h>
#include <iostream>

#include <cvrConfig/ConfigManager.h>
#include <OASClient.h>  // audio server
#include "BulletHandler.h"

using namespace std;
using namespace cvr;

int clnumavfs = 0;
std::vector<AntiGravityField*> clavfs;
int buttonState = 0, prevButtonState = -1;
AntiGravityField* hand;
btRigidBody* closest;
btVector3 initGrabPos;
btVector3 distToStylus;

extern ContactProcessedCallback gContactProcessedCallback;

CollisionType normalCollides = (CollisionType) (COL_NORMAL | COL_SPHERE);

void tickCallback(btDynamicsWorld *world, btScalar timeStep)
{
    // AVF
    for (int i = 0; i < clnumavfs; ++i) {
        //std::cout << "Num Ghost Collisions: " << clavfs[i]->getNumOverlappingObjects() << "\n";
        for(int j = 0; j < clavfs[i]->getNumOverlappingObjects(); j++) {
            btRigidBody *pRigidBody = dynamic_cast<btRigidBody *>(clavfs[i]->getOverlappingObject(j));
            pRigidBody->setGravity( clavfs[i]->getGravity() );
        }
    }
}

/* Using code from:
http://bulletphysics.org/Bullet/phpBB3/viewtopic.php?f=9&t=5483&view=next
*/
bool HandleContacts(btManifoldPoint& point, btCollisionObject* body0, btCollisionObject* body1)
{
   static struct timeval start, end;
   static bool veryFirst = true;
   static bool first = true;
   float timeThreshold = 350;
   float passedTime = 0;

   if (first)
   {
      first = false;
      if (veryFirst)
      {
         veryFirst = false;
         passedTime = timeThreshold;
      }
      gettimeofday(&start, NULL);
   }
   else
   {
      long mtime, seconds, useconds;
      gettimeofday(&end, NULL);
      seconds  = end.tv_sec  - start.tv_sec;
      useconds = end.tv_usec - start.tv_usec;

      mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
      passedTime = mtime;

      if (passedTime >= timeThreshold)
         first = true;
   }
   // ignore calls until enough time has passed.
   if (passedTime >= timeThreshold)
   {
//     cerr << "collision, " << point.getAppliedImpulse() << endl;
     float dist = point.getDistance();
     //cerr << "collision, " << dist << endl;
/*
     if (soundCollision!=NULL && soundCollision->isValid() && dist<-100.0)
     {
       soundCollision->setPosition(0, 0, 0);
       soundCollision->setDirection(0, 0, 0);     
       soundCollision->play();
     }
*/
   }

   return true;
}

BulletHandler::BulletHandler() 
{
    // Create World
    broadphase = new btDbvtBroadphase();
    btcc = new btDefaultCollisionConfiguration();
    btcd = new btCollisionDispatcher(btcc);
    btsolver = new btSequentialImpulseConstraintSolver;
    dynamicsWorld = new btDiscreteDynamicsWorld(btcd, broadphase, btsolver, btcc);
    dynamicsWorld->setGravity( btVector3(0,0,-10000) );
    
    // Add Floor
    /*
    btCollisionShape* groundShape = new btStaticPlaneShape(btVector3(0, 0, 1), 1);
    btDefaultMotionState* groundMotionState = new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), btVector3(0, 0, 0)));
    btRigidBody::btRigidBodyConstructionInfo groundRigidBodyCI(0, groundMotionState, groundShape, btVector3(0, 0, 0));
    btRigidBody* groundRigidBody = new btRigidBody(groundRigidBodyCI);
    dynamicsWorld->addRigidBody(groundRigidBody);
    */
    dynamicsWorld->setInternalTickCallback(tickCallback,this,true);
    dynamicsWorld->getBroadphase()->getOverlappingPairCache()->setInternalGhostPairCallback(new btGhostPairCallback());
    
    gContactProcessedCallback = (ContactProcessedCallback) HandleContacts;
    
    numRigidBodies = 0;
    numavfs = 0;
}

BulletHandler::~BulletHandler() {
    delete dynamicsWorld;
    delete btsolver;
    delete broadphase;
    delete btcc;
    delete btcd;
}

int BulletHandler::addBox( osg::Vec3 origin, osg::Vec3 halfLengths, osg::Quat quat, bool physEnabled ) {
    btCollisionShape* boxShape = new btBoxShape( *(btVector3*) &halfLengths );
    btDefaultMotionState* boxMotionState =
        new btDefaultMotionState(btTransform(btQuaternion(quat.x(), quat.y(), quat.z(), quat.w()), *(btVector3*)&origin));
        
    addRigid( boxShape, boxMotionState, COL_NORMAL, normalCollides, physEnabled );
    
    return numRigidBodies++;
}

int BulletHandler::addSeesaw( osg::Vec3 origin, osg::Vec3 halflengths, bool physEnabled ) {
    btCollisionShape* boxShape = new btBoxShape( *(btVector3*) &halflengths );
    btDefaultMotionState* boxMotionState =
        new btDefaultMotionState(btTransform(btQuaternion(0,0,0,1), *(btVector3*)&origin));
        
    btRigidBody* boxRigidBody = addRigid( boxShape, boxMotionState, COL_NORMAL, normalCollides, physEnabled );
    
    if (halflengths.x() > halflengths.y()) {
        btHingeConstraint* hinge = new btHingeConstraint(*boxRigidBody,btVector3(0,0,0),btVector3(0,1,0),true);
		    dynamicsWorld->addConstraint(hinge);
		} else {
        btHingeConstraint* hinge = new btHingeConstraint(*boxRigidBody,btVector3(0,0,0),btVector3(1,0,0),true);
		    dynamicsWorld->addConstraint(hinge);
		}
    
    return numRigidBodies++;
}

int BulletHandler::addOpenBox( osg::Vec3 origin, osg::Vec3 halfLengths, double innerWidth, bool physEnabled ) {
    btCollisionShape* xShape = new btBoxShape( btVector3(innerWidth/2, halfLengths.y(), halfLengths.z()) );
    btCollisionShape* yShape = new btBoxShape( btVector3(halfLengths.x(), innerWidth/2, halfLengths.z()) );
    btCollisionShape* zShape = new btBoxShape( btVector3(halfLengths.x(), halfLengths.y(), innerWidth/2) );
    
    btCompoundShape* boxShape = new btCompoundShape();
    
    boxShape->addChildShape( btTransform(btQuaternion(0,0,0,1), -btVector3(halfLengths.x() - innerWidth / 2, 0, 0)), xShape);
    boxShape->addChildShape( btTransform(btQuaternion(0,0,0,1), btVector3(halfLengths.x() - innerWidth / 2, 0, 0)), xShape);
    boxShape->addChildShape( btTransform(btQuaternion(0,0,0,1), -btVector3(0, halfLengths.y() - innerWidth / 2, 0)), yShape);
    boxShape->addChildShape( btTransform(btQuaternion(0,0,0,1), btVector3(0, halfLengths.y() - innerWidth / 2, 0)), yShape);
    boxShape->addChildShape( btTransform(btQuaternion(0,0,0,1), -btVector3(0, 0, halfLengths.z() - innerWidth / 2)), zShape);
    
    btDefaultMotionState* boxMotionState =
        new btDefaultMotionState(btTransform(btQuaternion(0,0,0,1), *(btVector3*) &origin));
        
    addRigid( boxShape, boxMotionState, COL_NORMAL, normalCollides, physEnabled );
    
    return numRigidBodies++;
}

int BulletHandler::addHollowBox( osg::Vec3 origin, osg::Vec3 halfLengths, bool physEnabled ) {
    const float innerWidth = 1.0f;
    btCollisionShape* xShape = new btBoxShape( btVector3(innerWidth/2, halfLengths.y(), halfLengths.z()) );
    btCollisionShape* yShape = new btBoxShape( btVector3(halfLengths.x(), innerWidth/2, halfLengths.z()) );
    btCollisionShape* zShape = new btBoxShape( btVector3(halfLengths.x(), halfLengths.y(), innerWidth/2) );
    
    btCompoundShape* boxShape = new btCompoundShape();
    
    boxShape->addChildShape( btTransform(btQuaternion(0,0,0,1), -btVector3(halfLengths.x() - innerWidth / 2, 0, 0)), xShape);
    boxShape->addChildShape( btTransform(btQuaternion(0,0,0,1), btVector3(halfLengths.x() - innerWidth / 2, 0, 0)), xShape);
    boxShape->addChildShape( btTransform(btQuaternion(0,0,0,1), -btVector3(0, halfLengths.y() - innerWidth / 2, 0)), yShape);
    boxShape->addChildShape( btTransform(btQuaternion(0,0,0,1), btVector3(0, halfLengths.y() - innerWidth / 2, 0)), yShape);
    boxShape->addChildShape( btTransform(btQuaternion(0,0,0,1), -btVector3(0, 0, halfLengths.z() - innerWidth / 2)), zShape);
    boxShape->addChildShape( btTransform(btQuaternion(0,0,0,1), btVector3(0, 0, halfLengths.z() - innerWidth / 2)), zShape);
    
    btDefaultMotionState* boxMotionState =
        new btDefaultMotionState(btTransform(btQuaternion(0,0,0,1), btVector3(origin.x(), origin.y(), origin.z())));
        
    btRigidBody * boxRigidBody = addRigid( boxShape, boxMotionState, COL_NORMAL, normalCollides, physEnabled );
    
    return numRigidBodies++;
}

int BulletHandler::addCustomObject( std::string path, std::vector<Triangle>* tris, double scale, osg::Vec3 pos, osg::Quat rot, bool physEnabled, float restitution ) {
    
    //btCollisionShape* triShape = meshBank[path];
    btCollisionShape* triShape = NULL;
    //if (triShape)
    //  std::cout << "Found collision mesh in cache." << std::endl;
    //else if (tris->size() == 0) return -1;
    if (tris->size() == 0)
    { 
        return -1;
    }
    else 
    {
      btTriangleMesh* tri_mesh = new btTriangleMesh();
      
      // add all the triangles to the mesh
      for (int i = 0; i < tris->size(); ++i) {
        //std::cout << tris->at(i).v1 << ", " << tris->at(i).v2 << ", " << tris->at(i).v3 << "\n";
        tri_mesh->addTriangle( *(btVector3*)&(tris->at(i).v1), *(btVector3*)&(tris->at(i).v2), *(btVector3*)&(tris->at(i).v3), true );
      }
      
      if (physEnabled) {
        triShape = new btConvexTriangleMeshShape( tri_mesh );
      } else {
        triShape = new btScaledBvhTriangleMeshShape( new btBvhTriangleMeshShape(tri_mesh, true), btVector3(scale,scale,scale) );
      }
      //meshBank[path] = triShape;
    }
    btQuaternion btq(rot.x(), rot.y(), rot.z(), rot.w());
    btDefaultMotionState* triMotionState =
        new btDefaultMotionState(btTransform(btq, *(btVector3*)&pos));
        
    btRigidBody * boxRigidBody = addRigid( triShape, triMotionState, COL_NORMAL, normalCollides, physEnabled, restitution );
    
    return numRigidBodies++;
}

int BulletHandler::addSphere( osg::Vec3 origin, double width, bool physEnabled ) {
    btCollisionShape* sphereShape = new btSphereShape( width );
    btDefaultMotionState* sphereMotionState =
        new btDefaultMotionState(btTransform(btQuaternion(0,0,0,1), *(btVector3*)&origin));
        
    addRigid( sphereShape, sphereMotionState, COL_SPHERE, (CollisionType) (normalCollides | COL_WALL), physEnabled );
    
    return numRigidBodies++;
}

int BulletHandler::addCylinder( osg::Vec3 origin, osg::Vec3 halfLengths, bool physEnabled ) {
    btCollisionShape* cylShape = new btCylinderShapeZ( *(btVector3*) &halfLengths );
    btDefaultMotionState* cylMotionState =
        new btDefaultMotionState(btTransform(btQuaternion(0,0,0,1), *(btVector3*)&origin));
        
    addRigid( cylShape, cylMotionState, COL_NORMAL, normalCollides, physEnabled );
    
    return numRigidBodies++;
}

void BulletHandler::addAntiGravityField(osg::Vec3 pos, osg::Vec3 halfLengths, osg::Vec3 grav) {
    btCollisionShape* ghostBox = new btBoxShape( *(btVector3*) &halfLengths );
    AntiGravityField* avf = new AntiGravityField();
    avf->setGravity( *(btVector3*) &grav );
    avf->setCollisionShape( ghostBox );
    avf->setCollisionFlags(avf->getCollisionFlags() | btCollisionObject::CF_NO_CONTACT_RESPONSE);
    avf->setWorldTransform( btTransform(btQuaternion(0,0,0,1), *(btVector3*) &pos) );
    dynamicsWorld->addCollisionObject(avf);
    
    avfs.push_back(avf);
    clavfs.push_back(avf);
    numavfs++;
    clnumavfs++;
}

void BulletHandler::addInvisibleWall( osg::Vec3 origin, osg::Vec3 halfLengths, int collisionFlag ) {
    btCollisionShape* boxShape = new btBoxShape( *(btVector3*) &halfLengths );
    btDefaultMotionState* boxMotionState =
        new btDefaultMotionState(btTransform(btQuaternion(0,0,0,1), *(btVector3*)&origin));
        
    addRigid( boxShape, boxMotionState, COL_WALL, (CollisionType) collisionFlag, false );
    rbodies.pop_back();
}

void BulletHandler::setLinearVelocity( int id, osg::Vec3 vel ) {
    rbodies[id]->setLinearVelocity( *(btVector3*) &vel );
}

void BulletHandler::translate( int id, osg::Vec3 vel ) {
    rbodies[id]->translate( *(btVector3*) &vel );
}

osg::Vec3 BulletHandler::getLinearVelocity( int id ) {
    btVector3 lv = rbodies[id]->getLinearVelocity();
    return *(osg::Vec3*) &lv;
}

void BulletHandler::stepSim( double lastFrame ) {
    dynamicsWorld->stepSimulation( lastFrame, 10, 1./80. );
}

void BulletHandler::getWorldTransform( int id, osg::Matrixd & boxm ) {
    btTransform m;
    if (id >= numRigidBodies) {
        boxm.makeIdentity();
        return;
    }
    btMotionState* ms = rbodies[id]->getMotionState();
    if (ms) {
        ms->getWorldTransform(m);
        boxm.setTrans( *(osg::Vec3*) &m.getOrigin() );
        btQuaternion btquat = rbodies[id]->getOrientation();
        osg::Quat osgquat(btquat.x(),btquat.y(),btquat.z(),btquat.w());
        boxm.setRotate( osgquat );
    } else boxm.makeIdentity();
}

void BulletHandler::setWorldTransform( int id, osg::Matrixd & boxm ) {
    btTransform m;
    if (id >= numRigidBodies) return;
    
    btMotionState* ms = rbodies[id]->getMotionState();
    if (ms) {
        osg::Vec3 t = boxm.getTrans();
        btVector3 btv( t.x(), t.y(), t.z() );
        osg::Quat q = boxm.getRotate();
        btQuaternion btq( q.x(), q.y(), q.z(), q.w() );
        btTransform btt( btq, btv);
        
        rbodies[id]->setCenterOfMassTransform(btt);
        ms->setWorldTransform(btt);
        //rbodies[id]->setGravity(btVector3(0,0,0));
        dynamicsWorld->synchronizeSingleMotionState( rbodies[id] );
    }
}

void BulletHandler::activate( int id ) {
  rbodies[id]->activate();
}

void BulletHandler::moveHand(osg::Matrixd & boxm ) {
  if (!hand) return;
  
  osg::Vec3 t = boxm.getTrans();
  btVector3 btv( t.x(), t.y(), t.z() );

  osg::Quat q = boxm.getRotate();
  btQuaternion bt90;
  bt90.setRotation( btVector3(1,0,0), 0.5 * 3.14159 / 180. );
  btQuaternion btq( q.x(), q.y(), q.z(), q.w() );
  btq *= bt90;
  btTransform btt(btq, btv);
  hand->setWorldTransform(btt);
  //hand->setInterpolationWorldTransform(btt);
  if (closest) {
    btMotionState* ms = closest->getMotionState();
    if (0) {
        closest->setCenterOfMassTransform(btt);
        ms->setWorldTransform(btt);
        closest->setLinearVelocity(btVector3(0,0,0));
        dynamicsWorld->synchronizeSingleMotionState( closest );
    }
  }
}

void BulletHandler::addHand(osg::Vec3 pos, osg::Vec3 halfLengths) {
    btCollisionShape* ghostBox = new btCylinderShapeZ( *(btVector3*) &halfLengths );
    AntiGravityField* avf = new AntiGravityField();
    avf->setGravity( btVector3(0,0,0) );
    avf->setCollisionShape( ghostBox );
    avf->setCollisionFlags(avf->getCollisionFlags() | btCollisionObject::CF_NO_CONTACT_RESPONSE);
    avf->setWorldTransform( btTransform(btQuaternion(0,0,0,1), *(btVector3*) &pos) );
    dynamicsWorld->addCollisionObject(avf);
    
    //if (hand) delete hand;
    hand = avf;
}

void BulletHandler::updateButtonState( int bs ) {
  prevButtonState = buttonState;
  buttonState = bs;
}

btDiscreteDynamicsWorld* BulletHandler::getDynamicsWorld() {
    return dynamicsWorld;
}

void BulletHandler::setGravity( osg::Vec3 g ) {
    dynamicsWorld->setGravity( *(btVector3*) &g );
}

btRigidBody* BulletHandler::addRigid( btCollisionShape* shape, btDefaultMotionState* ms, CollisionType collisionId, CollisionType collidesWith, bool physEnabled, float restitution ) {
    btRigidBody * _rb;
    
    if (physEnabled) {
        btVector3 inertia(0,0,0);
        shape->calculateLocalInertia(btScalar(1), inertia);
        btRigidBody::btRigidBodyConstructionInfo _rbci(1, ms, shape, inertia);
        _rbci.m_restitution = restitution;
        _rbci.m_linearSleepingThreshold = 20.0f;
        _rb = new btRigidBody(_rbci);
    } else {
        btRigidBody::btRigidBodyConstructionInfo _rbci(0, ms, shape, btVector3(0,0,0));
        _rbci.m_restitution = restitution;
        _rb = new btRigidBody(_rbci);
    }
    _rb->setCollisionFlags(_rb->getCollisionFlags() | btCollisionObject::CF_CUSTOM_MATERIAL_CALLBACK);
    
    dynamicsWorld->addRigidBody(_rb, collisionId, collidesWith);
    rbodies.push_back(_rb);
    
    return _rb;
}
