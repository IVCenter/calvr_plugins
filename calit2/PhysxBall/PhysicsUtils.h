#ifndef PHYSICS_UTILS
#define PHYSICS_UTILS

#include <foundation/PxVec3.h>
#include <osg/Vec3>
#include <osg/Matrix>

namespace osgPhysx
{
    inline physx::PxVec3 toPhysicsVec3( const osg::Vec3& v )
    { return physx::PxVec3(v[0], v[1], v[2]); }

    inline osg::Vec3 toVec3( const physx::PxVec3& v )
    { return osg::Vec3(v[0], v[1], v[2]); }

    extern osg::Matrix physX2OSG_Rotation( const physx::PxMat44& m );

    extern physx::PxMat44 toPhysicsMatrix( const osg::Matrix& matrix );
    extern osg::Matrix toMatrix( const physx::PxMat44& pmatrix );
}

#endif
