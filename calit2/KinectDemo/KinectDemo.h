#ifndef _KINECTDEMO_
#define _KINECTDEMO_

#include <osgDB/ReadFile>
#include <osgDB/FileUtils>
#include <osg/PolygonMode>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/SceneObject.h>
#include <cvrKernel/ScreenConfig.h>
#include <cvrConfig/ConfigManager.h>
#include <cvrInput/TrackingManager.h>
#include <cvrInput/TrackerBase.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/CVRPlugin.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuRangeValue.h>
#include <cvrMenu/MenuRangeValueCompact.h>
#include <cvrMenu/TabbedDialogPanel.h>
#include <cvrKernel/ComController.h>

//#include <cvrInput/TrackerPlugin.h>
#include "Skeleton.h"
//#include "PubSub.h"
#include <shared/PubSub.h>
#include <protocol/skeletonframe.pb.h>
#include <protocol/colormap.pb.h>
#include <protocol/depthmap.pb.h>
#include <protocol/pointcloud.pb.h>
#include <zmq.hpp>
#include "kUtils.h"
#include "SelectableItem.h"
#include "CloudManager.h"
#include "KinectObject.h"
//#include "SkeletonManager.h"
#include <unordered_map>

using namespace std;
using namespace osg;
using namespace cvr;

#define M_HEAD 1
#define M_LHAND 9
#define M_RHAND 15
#define M_LFOOT 20
#define M_RFOOT 24


//std::map<int, Skeleton> mapIdSkel;
std::unordered_map<float, osg::Vec4f> distanceColorMap;
std::unordered_map<float, osg::Vec4f> distanceColorMapDepth;
std::unordered_map<float, uint32_t> dpmap;
//uint32_t dpmap[15000];


zmq::context_t context(1);
SubSocket<RemoteKinect::SkeletonFrame>* skel_socket;
SubSocket<RemoteKinect::DepthMap>* depth_socket;
SubSocket<RemoteKinect::PointCloud>* cloud_socket;
SubSocket<RemoteKinect::PointCloud>* cloud_socket2;
SubSocket<RemoteKinect::ColorMap>* color_socket;

//zmq::context_t contextCloud(1);
//SubSocket<RemoteKinect::PointCloud>* cloudT_socket;

uint32_t color_pixels[480 * 640];
uint32_t depth_pixels[640 * 480];

static osg::ref_ptr<osg::Geode> pointGeode;

int navLock = -1;


class KinectDemo : public cvr::MenuCallback, public cvr::CVRPlugin
{
public:
    KinectDemo();
    virtual ~KinectDemo();

    bool init();

    void menuCallback(cvr::MenuItem* item);
    bool processEvent(InteractionEvent* event);
    void preFrame();

    static KinectDemo* instance();

    int bcounter;
    float colorfps;

    bool useKColor;

    bool useKinect;
    bool kinectInitialized;

    bool kShowColor;
    bool kShowPCloud;
    bool kinectThreaded;
    bool _firstRun;
    bool skeletonThreaded;
    bool kNavSpheres;
    bool kShowDepth;
    bool kMoveWithCam;
    bool kFreezeCloud;
    bool kShowArtifactPanel;
    bool kShowInfoPanel;
    osg::Program* pgm1;
    osg::ref_ptr<osg::Group> kinectgrp;
    osg::Geometry* knodeGeom;
    //    osg::Geometry* kinectGeom;
    //    osg::DrawArrays* kinectDrawArrays;
    osg::Group* kinectgrp2;
    //    osg::Geometry* kinectgeo2;
    //    osg::DrawArrays* kinectDrawArrays2;
    float initialPointScale;

    std::vector<KinectObject*>* kinects;

    struct PointCloud
    {
        string name;
        string filename;
        string fullpath;
        string filetype;
        string modelType;
        string group;
        float scale;
        osg::Vec3 pos;
        osg::Quat rot;
        osg::Vec3 origPos;
        osg::Quat origRot;
        float origScale;
        cvr::SceneObject* so;
        bool loaded;
        bool active;
        bool visible;
        bool selected;
        bool lockPos;
        bool lockRot;
        bool lockScale;
        bool lockGraph;
        int lockedTo;
        int lockedType;
        cvr::MenuButton* saveMap;
        cvr::MenuButton* saveNewMap;
        cvr::MenuButton* resetMap;
        cvr::MenuCheckbox* activeMap;
        cvr::MenuCheckbox* visibleMap;
        cvr::MenuRangeValue* rxMap;
        cvr::MenuRangeValue* ryMap;
        cvr::MenuRangeValue* rzMap;
        //Store Different Model Type Transforms
        osg::Node* currentModelNode;
        osg::Switch* switchNode;


    };

    std::vector<PointCloud* > _pointClouds;
    std::vector<SelectableItem> selectableItems;
    void createSceneObject();
    void createSceneObject2();
    void sendEvents();
    void kinectInit();
    void kinectOff();
    void moveCam(double, double, double, double, double, double, double, double);
    void createSelObj(osg::Vec3 pos, std::string, float radius, osg::Node* model);

    void ThirdInit();
    void ThirdLoop();
    void updateInfoPanel();
    void ExportPointCloud();
    RemoteKinect::SkeletonFrame* sf;
    RemoteKinect::PointCloud* packet;
    RemoteKinect::ColorMap* cm;
    RemoteKinect::DepthMap* dm;
    int minDistHSV, maxDistHSV;
    int minDistHSVDepth, maxDistHSVDepth;
    osg::Vec4f getColorRGB(int dist);
    osg::Vec4f getColorRGBDepth(int dist);

    void cloudOff();
    void navOff();
    void navOn();
    void cloudOn();
    void colorOff();
    void colorOn();
    void depthOff();
    void depthOn();
    void moveWithCamOff();
    void moveWithCamOn();
    void checkHandsIntersections(int skel_id, std::map<int, Skeleton>* skel_map);

    void showCameraImage();
    void showDepthImage();

    void saveEnvironment();

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

    osg::ref_ptr<osg::Vec4Array> kinectColours;
    osg::ref_ptr<osg::Vec3Array> kinectVertices;
    osg::ref_ptr<osg::Vec4Array> kinectColours2;
    osg::ref_ptr<osg::Vec3Array> kinectVertices2;
    osg::ref_ptr<osg::Geode> kgeode;
    CloudManager* _cloudThread;
    // SkeletonManager * _skeletonThread;
protected:

    static KinectDemo* _kinectDemo;

    cvr::SubMenu* _avMenu;
    cvr::MenuCheckbox* _kColorOn;
    cvr::MenuCheckbox* _kShowColor;
    cvr::MenuCheckbox* _kShowPCloud;
    cvr::MenuCheckbox* _kMoveWithCam;
    cvr::MenuCheckbox* _kFreezeCloud;
    cvr::MenuCheckbox* _kinectOn;
    cvr::MenuRangeValue* _kColorFPS;
    cvr::MenuCheckbox* _kNavSpheres;
    cvr::MenuCheckbox* _kShowDepth;
    cvr::MenuCheckbox* _kShowInfoPanel;
    cvr::TabbedDialogPanel* _infoPanel;
    cvr::MenuRangeValue* sliderX;
    cvr::MenuRangeValue* sliderY;
    cvr::MenuRangeValue* sliderZ;
    cvr::MenuRangeValue* slider2X;
    cvr::MenuRangeValue* slider2Y;
    cvr::MenuRangeValue* slider2Z;
    cvr::MenuRangeValue* sliderRX;
    cvr::MenuRangeValue* sliderRY;
    cvr::MenuRangeValue* sliderRZ;
    cvr::MenuRangeValue* sliderRW;

    int NPIX;
    int NFRAMES;
    float* X;
    float* Y;
    float* Z;
    float* environmentX;
    float* environmentY;
    float* environmentZ;
    float** frameXMinus;
    float** frameYMinus;
    float** frameZMinus;
    cvr::SubMenu* _devMenu;
    cvr::MenuCheckbox* _devFixXY;
    bool fixXY;
    cvr::MenuCheckbox* _devIgnoreZeros;
    bool ignoreZeros;
    cvr::MenuCheckbox* _devFilterBackground;
    bool filterBackground;
    cvr::MenuCheckbox* _devAssignPointsToSkeletons;
    bool assignPointsToSkeletons;
    cvr::MenuCheckbox* _devClassifyPoints;
    bool classifyPoints;
    cvr::MenuCheckbox* _devDenoise;
    bool denoise;

    void showPointCloud();

    float distanceMIN, distanceMAX;
    cvr::MenuButton* _bookmarkLoc;
    cvr::MenuButton* _testInteract;
    osg::MatrixTransform* _root;


    osg::Node* _modelFileNode1;
    osg::Node* _modelFileNode2;
    osg::Node* _modelFileNode3;
    osg::Node* _modelFileNode4;
    osg::Node* _modelFileNode5;

    float _sphereRadius;

    float kinectX;
    float kinectY;
    float kinectZ;
    float kinect2X;
    float kinect2Y;
    float kinect2Z;
    float kinectRX;
    float kinectRY;
    float kinectRZ;
    float kinectRW;
    float kinect2RX;
    float kinect2RY;
    float kinect2RZ;
    float kinect2RW;

};

#endif
