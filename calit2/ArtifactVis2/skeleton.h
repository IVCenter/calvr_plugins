#include <osg/Material>
#include <osg/MatrixTransform>
#include <osgText/Text>
#include <osg/Drawable>

#include <osg/ShapeDrawable>
#include <osg/Drawable>
#include <osg/Vec2>
#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/Array>
#include <osg/PrimitiveSet>

#include "MCylinder.h"

//#include <opencv/highgui.h>

// following to eliminate some errors in opencv
#ifdef True
#undef True
#endif

#ifdef False
#undef False
#endif

//#include <opencv/cv.h>

#define ROI_OFFSET 70 // region of interest size




static osg::Vec4 _colorsJoints[729];
static int colorsInitialized = 0;


struct JointNode
{

    osg::Quat   q;
    //    osg::Vec3d  campos;
    double      scale;

    //    osg::MatrixTransform* camt;
    //    osg::MatrixTransform* camrt;

    osg::MatrixTransform*   translate;
    osg::MatrixTransform*   rotate;
    osg::Geode*             geode;

    // projective coordinates for hands
    double image_x;
    double image_y;

    JointNode();
    int id;
    osg::Vec3d position;
    osg::Vec3d prevPosition;
    double orientation[4];

    void update(int joint_id, osg::Vec3d pos, double ori[4], bool attached, bool lHandOpen, bool rHandOpen);
    //    void updateOrient(int joint_id, double ori[4]);
    void makeDrawable(int i);
    osg::Vec4 getColor(std::string dc);

    //    void setcampos(osg::Vec3d _campos, osg::Quat _q, double scale);
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

/*
struct ConvexityDefect
{
    cv::Point start;
    cv::Point end;
    cv::Point depth_point;
    float depth;
};
*/


struct Skeleton
{
    static bool moveWithCam;
    static osg::Vec3d camPos;
    static osg::Quat camRot;
    static bool navSpheres;

    int YRES;// = 480;
    int XRES;// = 640;
    int ROIOFFSET;// = 70;
    float DEPTH_SCALE_FACTOR;// = 255./4096.;
    unsigned int BIN_THRESH_OFFSET;// = 5;
    unsigned int MEDIAN_BLUR_K;// = 5;
    double GRASPING_THRESH;// = 0.9;
    int HAND_SIZE;// = 500; // originally 2000

    bool lHandOpen;
    bool rHandOpen;

    bool attached;
    MCylinder bone[15];
    MCylinder cylinder;
    NavigationSphere navSphere;
    Skeleton();
    osg::Vec3 offset;
    JointNode joints[25];
    void update(int joint_id, osg::Vec3d pos, double ori[4]);
    //    void updateOrient(int joint_id, double ori[4]);
    void addOffset(osg::Vec3);
    void attach(osg::MatrixTransform* parent);
    void detach(osg::MatrixTransform* parent);
    //  cv::Rect roi;

    // void checkHandOpen(int handId, cv::Mat handMat, int handDepth, int ROIOFFSET);


    // void findConvexityDefects(std::vector<cv::Point>& contour, std::vector<int>& hull, std::vector<ConvexityDefect>& convexDefects);
    //    void setcampos(osg::Vec3d _campos, osg::Quat _q, double scale);

};

