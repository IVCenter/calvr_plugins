#include "PhysicsUtils.h"
#include <PxPhysicsAPI.h>

using namespace osg;
using namespace physx;
using namespace osgPhysx;
namespace osgPhysx {
osg::Matrix physX2OSG_Rotation(const PxMat44 &m) {
    PxVec3 pos = m.getPosition();
    Matrix trans;
    trans.makeTranslate(pos.x, -pos.z, pos.y);

    Matrix rotMat;
    double w = sqrt(1.0 + m(0, 0) + m(1, 1) + m(2, 2)) / 2.0;
    double w4 = (4.0 * w);
    double x = (m(2, 1) - m(1, 2)) / w4;
    double y = (m(0, 2) - m(2, 0)) / w4;
    double z = (m(1, 0) - m(0, 1)) / w4;
    rotMat.makeRotate(Quat(x, -z, y, w));
    return trans;
}

    PxMat44 toPhysicsMatrix( const osg::Matrix& matrix )
    {
        PxReal d[16];
        for ( int i=0; i<16; ++i ) d[i] = *(matrix.ptr() + i);
        return PxMat44(d);
    }

    osg::Matrix toMatrix( const PxMat44& pmatrix )
    {
        double m[16];
        for ( int i=0; i<16; ++i ) m[i] = *(pmatrix.front() + i);
        return osg::Matrix(&m[0]);
    }

    Matrix glm_to_osg(glm::mat4 glm_mat){
        Matrix osg_mat;
        glm::quat q = glm::toQuat(glm_mat);
        glm::vec3 t(glm_mat[3][0],glm_mat[3][1],glm_mat[3][2]);
        osg::Quat oq(q.x, q.y, q.z, q.w);
        Matrix osgt = Matrix::translate(t.x,t.y,t.z);
        Matrix osgr = Matrix::rotate(oq);
        osg_mat = osgt * osgr;
        return osg_mat;
    }
}