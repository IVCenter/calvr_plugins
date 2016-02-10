#ifndef _ANTIGRAVFIELD_H
#define _ANTIGRAVFIELD_H

#include <BulletCollision/CollisionDispatch/btGhostObject.h>

class AntiGravityField : public btPairCachingGhostObject {
  public:
    void setGravity( btVector3 g ) { grav = g; }
    btVector3 getGravity() { return grav; }
    
  private:
    btVector3 grav;
};
#endif
