#include <osg/Material>
#include <osg/MatrixTransform>
#include <osgText/Text>

struct MCylinder
{
    MCylinder();
    osg::Vec3d currVec;
    osg::Vec3d prevVec;
    osg::Vec3 StartPoint;
    osg::Vec3 EndPoint;
    osg::Vec3 center;
    osg::MatrixTransform*   translate;
    osg::Quat rotation;
    osg::Geode*             geode;
    bool attached;
    bool locked;
    void update(osg::Vec3 startP, osg::Vec3 endP);
    void attach(osg::MatrixTransform* parent);
    void detach(osg::MatrixTransform* parent);
    float radius;
    osg::Vec4 CylinderColor;
    MCylinder(float _radius, osg::Vec4 _color);

    double length;
    double prevLength;

    bool handsBeenAboveElbows;
};
