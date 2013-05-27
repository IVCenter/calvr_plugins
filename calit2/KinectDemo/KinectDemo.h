#ifndef _KINECTDEMO_
#define _KINECTDEMO_

#include <osgDB/ReadFile>
#include <osgDB/FileUtils>
#include <osg/PolygonMode>
#include <osg/Matrix>
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

#include "Skeleton.h"
#include "kUtils.h"
#include "SelectableItem.h"
#include "InputManager.h"
#include "KinectInteractions.h"
#include "CloudManager.h"
#include "KinectObject.h"
#include <unordered_map>

using namespace std;
using namespace osg;
using namespace cvr;

#define M_HEAD 1
#define M_LHAND 15
#define M_RHAND 9
#define M_LFOOT 24
#define M_RFOOT 20

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
    InputManager* inputManager;
    KinectInteractions* kinectInteractions;
    int bcounter;
    int masterKinect;
    int oldMasterKinect;
    int max_users;
    float colorfps;

    bool useKColor;
    bool userColor;

    bool useKinect;
    bool kinectInitialized;

    bool kShowColor;
    bool kShowPCloud;
    bool kinectThreaded;
    bool _firstRun;
    bool skeletonThreaded;
    bool kShowDepth;
    bool kMoveWithCam;
    bool kFreezeCloud;
    bool kShowArtifactPanel;
    bool kShowInfoPanel;
    float initialPointScale;

    std::vector<KinectObject*>* kinects;

    std::vector<SelectableItem> selectableItems;
    //    void createSceneObject();
    //    void createSceneObject2();
    void sendEvents();
    void kinectInit();
    void kinectOff();
    void moveCam(double, double, double, double, double, double, double, double);
    void createSelObj(osg::Vec3 pos, std::string, float radius, osg::Node* model);

    void updateInfoPanel();
    //    void ExportPointCloud();

    void cloudOff();
    void cloudOn();
    void colorOff();
    void colorOn();
    void depthOff();
    void depthOn();
    void moveWithCamOff();
    void moveWithCamOn();
    void checkHandsIntersections(int skel_id, std::map<int, Skeleton>* skel_map);
    void loadScreensMenu();
    void checkSkelMaster(std::map<int, Skeleton>* skel_map);
    void checkSkelGesture(std::map<int, Skeleton>* skel_map);
    void gestureSurrender(osg::Vec3 lHand,osg::Vec3 rHand,osg::Vec3 head);
    void gestureLeftClick(osg::Vec3 lHand,osg::Vec3 rHand,osg::Vec3 head);
    void gestureRightClick(osg::Vec3 lHand,osg::Vec3 rHand,osg::Vec3 head);
    //    void kinectTransform();

    //    osg::ref_ptr<osg::Geode> kgeode;
    //    CloudManager* _cloudThread;
    CalibrateKinect* _calibraterTool;

    //Helmert Global Variables
    //    std::vector<osg::Vec3> helmertTArray;
    //    std::vector<osg::Matrix> helmertMArray;
    //    std::vector<float> helmertSArray;
    bool wandLockedToSkeleton;
protected:

    static KinectDemo* _kinectDemo;

    cvr::SubMenu* _avMenu;
    cvr::MenuCheckbox* _kColorOn;
    cvr::MenuCheckbox* _kUserColorOn;
    cvr::MenuCheckbox* _kShowColor;
    cvr::MenuCheckbox* _kShowPCloud;
    cvr::MenuCheckbox* _kMoveWithCam;
    cvr::MenuCheckbox* _kFreezeCloud;
    cvr::MenuCheckbox* _kinectOn;
    cvr::MenuRangeValue* _kColorFPS;
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
    cvr::MenuButton* _switchMasterSkeleton;

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

    cvr::SubMenu* _calibrateMenu;
    cvr::SubMenu* _calibrateIrMenu;
    cvr::SubMenu* _calibrateRefMenu;
    cvr::MenuCheckbox* _toggleCalibrate;
    cvr::MenuCheckbox* _toggleRefCalibrate;
    cvr::MenuButton* _skeletonCalibrate;
    cvr::MenuCheckbox* _kinectPC1;
    cvr::MenuCheckbox* _kinectPC2;
    cvr::MenuCheckbox* _kinectPC3;
    cvr::MenuCheckbox* _kinectPC4;
    cvr::MenuCheckbox* _kinectRef;
    cvr::MenuCheckbox* _kinectTransformed;
    cvr::MenuCheckbox* _kinectIrTransformed;
    cvr::MenuCheckbox* _kinectCreateIrPoint;
    cvr::MenuCheckbox* _kinectCreateSelectPoint;
    cvr::MenuButton* _eraseAllSelectPoints;
    cvr::MenuButton* _triangulateKinect;
    cvr::MenuButton* _triangulateIrKinect;
    cvr::MenuCheckbox* _toggleButton0;
    cvr::MenuCheckbox* _toggleNavigation;
    cvr::MenuCheckbox* _showRefPoints;
    cvr::MenuCheckbox* _showIRPoints;
    std::vector<cvr::MenuCheckbox*> screen_list;
    std::vector<std::string> screen_path;
    PointsLoadInfo _kinectCloud1;
    PointsLoadInfo _kinectCloud2;
    PointsLoadInfo _kinectCloud3;
    PointsLoadInfo _kinectCloud4;
    PointsLoadInfo _kinectCloudRef;
    PointsLoadInfo _kinectCloudTransformed;
    PointsLoadInfo _kinectCloudIrTransformed;
    std::vector<osg::Geode*> selectPoints;
    std::vector<osg::ref_ptr<osg::Group> > screenGroup;
    osg::Group* refPointsGroup;
    osg::Group* irPointsGroup;
    bool buttonDown;
    bool rightButtonDown;
    
    int  calibCount;

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
