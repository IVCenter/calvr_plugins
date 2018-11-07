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
}