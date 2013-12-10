#include "CloudManager.h"
#include "SkeletonManager.h"
#include <osg/PolygonMode>
#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/SceneObject.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuRangeValue.h>
#include <cvrMenu/MenuCheckbox.h>
#include "Skeleton.h"
#include <cvrKernel/PluginHelper.h>
#include <zmq.hpp>
#include "CalibrateKinect.h"
#include <dirent.h>
#include <sys/stat.h>

class KinectObject : public cvr::SceneObject
{
public:
    KinectObject(std::string name, std::string cloud_server, std::string skeleton_server, std::string color_server, std::string depth_server, osg::Vec3 position);
    CloudManager* cm;
    SkeletonManager* sm;
    //osg::Group* group;
    //osg::Geometry* geom;
    //osg::Geode* kgeode;
    std::vector<osg::ref_ptr<osg::Group> > _cloudGroups;
    osg::Switch* switchNode;
    osg::Program* pgm1;
    osg::Node* _modelFileNode1;
    //osg::DrawArrays* drawArray;
    float initialPointScale;
    bool _firstRun;
    bool _cloudIsOn;
    bool _navigatable;
    bool useHands;
    bool _kinectFOV_on;
    std::string _kinectName;

    int max_users;
    float kinectX;
    float kinectY;
    float kinectZ;
    std::string cloudServer;
    std::string skeletonServer;
    std::string colorServer;
    std::string depthServer;

    RemoteKinect::SkeletonFrame* skel_frame;
    SubSocket<RemoteKinect::SkeletonFrame>* skel_socket;
    std::map<int, Skeleton> mapIdSkel;

    //osg::ref_ptr<osg::Vec4Array> kinectColours;
    //osg::ref_ptr<osg::Vec3Array> kinectVertices;
    osg::Vec4Array* kinectColours;
    osg::Vec3Array* kinectVertices;

    void setupMenus();
    void loadTransformMenu();
    void menuCallback(cvr::MenuItem* item);
    std::vector<cvr::MenuButton*>* transform_list;
    std::vector<cvr::MenuCheckbox*> _toggleUsersArray;
    cvr::MenuCheckbox* _toggleKinectFOV;
    std::vector<std::string>* transform_path;
    void transformFromFile(std::string filepath);
    std::vector<osg::Vec3> helmertTArray;
    std::vector<osg::Matrix> helmertMArray;
    std::vector<float> helmertSArray;

    void cloudInit();
    void cloudOn();
    void cloudUpdate();
    void cloudOff();

    void skeletonOn();
    void skeletonUpdate();
    void skeletonOff();
    std::map<int, Skeleton>* skeletonGetMap();

    void cameraOn();
    void cameraUpdate();
    void cameraOff();
    void depthOn();
    void depthUpdate();
    void depthOff();
    void toggleNavigation(bool navigatable);
    void showKinectFOV();

    // camera and depth image

    int minDistHSVDepth, maxDistHSVDepth;
    std::unordered_map<float, osg::Vec4f> distanceColorMapDepth;
    RemoteKinect::ColorMap* colm;
    RemoteKinect::DepthMap* depm;
    osg::Vec4f getColorRGBDepth(int dist);
    SubSocket<RemoteKinect::DepthMap>* depth_socket;
    SubSocket<RemoteKinect::ColorMap>* color_socket;
    std::unordered_map<float, uint32_t> dpmap;
    bool _cameraOn;
    bool _depthOn;
    uint32_t color_pixels[480 * 640];
    uint32_t depth_pixels[640 * 480];
    //camera image things
    osg::ref_ptr<osg::MatrixTransform> bitmaptransform;
    osg::ref_ptr<osg::Image> image;
    osg::Texture2D* pTex;
    osg::Geode* pGeode;
    osg::StateSet* pStateSet;
    osg::ref_ptr<osg::Geometry> geometry;
    osg::ref_ptr<osg::Vec3Array> vertexArray;
    osg::ref_ptr<osg::Vec4Array> colorArray;
    osg::ref_ptr<osg::Vec2Array> texCoordArray;
    //depth sensor things
    osg::ref_ptr<osg::MatrixTransform> depthBitmaptransform;
    osg::ref_ptr<osg::Image>    depthImage;
    osg::Texture2D*             depthPTex;
    osg::Geode*                 depthPGeode;
    osg::StateSet*              depthPStateSet;
    osg::ref_ptr<osg::Geometry> depthGeometry;
    osg::ref_ptr<osg::Vec3Array>depthVertexArray;
    osg::ref_ptr<osg::Vec4Array>depthColorArray;
    osg::ref_ptr<osg::Vec2Array>depthTexCoordArray;
    osg::ref_ptr<osg::MatrixTransform> kinectFOV;


    zmq::context_t* context;
protected:

};
