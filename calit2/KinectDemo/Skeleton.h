
#include "MCylinder.h"
#include <osg/ShapeDrawable>
#include <cvrKernel/SceneObject.h>
#include "kUtils.h"

static bool colorsInitialized;
static osg::Vec4 _colors[729];
struct JointNode
{
    osg::Quat   q;

    osg::MatrixTransform*   translate;
    osg::MatrixTransform*   rotate;
    osg::Geode*             geode;

    JointNode();
    int id;
    osg::Vec3d position;
    double orientation[4];

    void update(int joint_id, float newx, float newy, float newz, float neworx, float newory, float neworz, float neworw, bool attached);
    void makeDrawable(int i);

    osg::Vec4 getColor(std::string dc);
};

struct NavigationSphere
{
    osg::MatrixTransform*   translate;
    osg::MatrixTransform*   rotate;
    osg::Geode*             geode;

    osg::Vec3 position;
    osg::Vec3 prevPosition;

    int lock;
    bool activated;
    NavigationSphere();

    void update(osg::Vec3d position2, osg::Vec4f orientation);
};

struct Skeleton
{
    static bool moveWithCam;
    static osg::Vec3d camPos;
    static osg::Quat camRot;
    static bool navSpheres;

    // are hands holding objects?
    bool leftHandBusy;
    bool rightHandBusy;

    bool attached;
    MCylinder bone[15];
    MCylinder cylinder;
    NavigationSphere navSphere;
    Skeleton();
    JointNode joints[25];
    void update(int joint_id, float newx, float newy, float newz, float neworx, float newory, float neworz, float neworw);
    void attach(osg::Switch* parent);
    void detach(osg::Switch* parent);
};

