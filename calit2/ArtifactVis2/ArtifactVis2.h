#ifndef _ARTIFACTVIS2_
#define _ARTIFACTVIS2_

#include <cvrKernel/CVRPlugin.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/TabbedDialogPanel.h>
#include <cvrMenu/DialogPanel.h>
#include <cvrMenu/MenuRangeValue.h>
#include <cvrMenu/MenuTextButtonSet.h>
#include <cvrMenu/MenuList.h>

#include <osg/Material>
#include <osg/MatrixTransform>
#include <osgText/Text>

#include <string>
#include <vector>
#include <map>

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <X11/Xlib.h>
#include <spnav.h>

#include "skeleton.h"
#include <shared/PubSub.h>
#include <protocol/skeletonframe.pb.h>
#include <protocol/colormap.pb.h>
#include <protocol/depthmap.pb.h>
#include <protocol/pointcloud.pb.h>
#include <zmq.hpp>


#define M_HEAD 1
#define M_LHAND 9
#define M_RHAND 15
#define M_LFOOT 20
#define M_RFOOT 24
#define CYLINDER 50
//cv::Mat depthRaw(480, 640, CV_16UC1);
const float DEPTH_SCALE_FACTOR = 255. / 4096.;
std::map<int, Skeleton> mapIdSkel;

std::map< std::string, osg::ref_ptr<osg::Node> > objectMap;

zmq::context_t context(1);
SubSocket<RemoteKinect::SkeletonFrame>* skel_socket;

uint32_t color_pixels[480 * 640];
SubSocket<RemoteKinect::ColorMap>* color_socket;

float depth_pixels[640 * 480];
SubSocket<RemoteKinect::DepthMap>* depth_socket;

SubSocket<RemoteKinect::PointCloud>* cloud_socket;
static osg::ref_ptr<osg::Geode> pointGeode;

// move somewhere else?
struct SelectableItem
{
    osg::ref_ptr<osg::MatrixTransform> scalet;
    osg::ref_ptr<osg::Geode> boxGeode;
    osg::ref_ptr<osg::MatrixTransform> mt;
    osg::ref_ptr<osg::MatrixTransform> rt;
    double scale;
    osg::Vec3 position;
    int lock;
    int type;
    SelectableItem()
    {
        mt = new osg::MatrixTransform();
        lock = -1;
        type = -1;
    }
    SelectableItem(osg::ref_ptr<osg::Geode> g, osg::ref_ptr<osg::MatrixTransform> _model, osg::ref_ptr<osg::MatrixTransform> m, osg::ref_ptr<osg::MatrixTransform> r, double _scale)
    {
        scalet = _model;
        boxGeode = g;
        mt = m;
        lock = -1;
        type = -1;
        rt = r;
        scale = _scale;
    }

    void setScale(double s)
    {
        scale = s;
        osg::Matrixd scalem;
        scalem.makeScale(s, s, s);
        scalet->setMatrix(scalem);
    }

    void lockTo(int lockedTo, int lockType)
    {
        lock = lockedTo;
        type = lockType;
        rt->removeChild(0, 1);
        rt->addChild(scalet);
    }

    void unlock()
    {
        lock = -1;
	type = -1;
        rt->removeChild(0, 1);
        //        rt->addChild(boxGeode); // change back to box when not manipulated -- with timeout? never?
        rt->addChild(scalet);
    }

};
int navLock =-1;
struct NavItem
{
    osg::ref_ptr<osg::MatrixTransform> scalet;
    osg::ref_ptr<osg::Geode> boxGeode;
    osg::ref_ptr<osg::MatrixTransform> mt;
    osg::ref_ptr<osg::MatrixTransform> rt;
    double scale;
    osg::Vec3 position;
    int lock;
    NavItem()
    {
        mt = new osg::MatrixTransform();
        lock = -1;
    }
    NavItem(osg::ref_ptr<osg::Geode> g, osg::ref_ptr<osg::MatrixTransform> _model, osg::ref_ptr<osg::MatrixTransform> m, osg::ref_ptr<osg::MatrixTransform> r, double _scale)
    {
        scalet = _model;
        boxGeode = g;
        mt = m;
        lock = -1;
        rt = r;
        scale = _scale;
    }

    void setScale(double s)
    {
        scale = s;
        osg::Matrixd scalem;
        scalem.makeScale(s, s, s);
        scalet->setMatrix(scalem);
    }

    void lockTo(int lockedTo)
    {
        lock = lockedTo;
        rt->removeChild(0, 1);
        rt->addChild(scalet);
    }

    void unlock()
    {
        lock = -1;
        rt->removeChild(0, 1);
        //        rt->addChild(boxGeode); // change back to box when not manipulated -- with timeout? never?
        rt->addChild(scalet);
    }

};
class Artifact
{
public:
    std::string dc;
    std::vector<std::string> fields;
    std::vector<std::string> values;
    double pos[3];
    osg::Drawable* drawable;
    osgText::Text* label;
    osg::Vec3 modelPos;
    osg::Vec3 modelOriginalPos;
    bool visible;
    bool selected;
    bool showLabel;
    double distToCam;
  
    osg::ref_ptr<osg::PositionAttitudeTransform> patmt;
    osg::ref_ptr<osg::MatrixTransform> rt;
    osg::ref_ptr<osg::MatrixTransform> scalet;
    int lockedTo;
    int lockedType;
    osg::Vec3d kpos;
    double scale;

    Artifact()
    {
        pos[0] = 0.0;
        pos[1] = 0.0;
        pos[2] = 0.0;
        showLabel = true;
    }

    void setScale(double s)
    {
        scale = s;
        osg::Matrixd scalem;
        scalem.makeScale(s, s, s);
        scalet->setMatrix(scalem);
    }

};


class ArtifactVis2 : public cvr::MenuCallback, public cvr::CVRPlugin
{
public:
    ArtifactVis2();
    virtual ~ArtifactVis2();

    class compare
    {
    public:
        bool operator()(const Artifact* a1, const Artifact* a2)
        {
            return ((*a1).distToCam < (*a2).distToCam);
        }
    };
    void message(int type, char* data);

    bool init();

    bool processEvent(cvr::InteractionEvent* event);
    bool statusSpnav;
    bool _handOn;
    void menuCallback(cvr::MenuItem* item);
    void preFrame();

    void setDCVisibleStatus(std::string dc, bool status);
    void updateVisibleStatus();
    std::string parseDate(std::string date);
    static ArtifactVis2* instance();
    std::vector<osg::Vec3> getArtifactsPos();
    float selectArtifactSelected();
    osg::Matrix getSelectMatrix();
    void setSelectMatrix(osg::Matrix& mat);
    bool _selectActive;

    int _activeArtifact;

    void testSelected();

    //Kinect
    NavigationSphere navSphere;
	    double navSphereOffsetY;
	    double navSphereOffsetZ;
	    //X
            double diffScaleX;
	    double diffScaleNegX;
            double tranScaleX;
	    //Y
            double diffScaleY;
	    double diffScaleNegY;
            double tranScaleY;
	    //Z
            double diffScaleZ;
	    double diffScaleNegZ;
            double tranScaleZ;

	    //RY
            double diffScaleRY;
	    double diffScaleNegRY;
            double tranScaleRY;

    osg::ref_ptr<osg::MatrixTransform> bitmaptransform;
    int bcounter;
    float colorfps;
    float navSphereTimer;
    int kinectUsers;
    bool handsBeenAbove;
    bool navSphereActivated;
    bool useKColor;
    bool useKinect;
    bool kSelectKinect;
    bool kShowColor;
    bool kShowPCloud;
    bool kNavSpheres;
    bool kUseGestures;
    bool kMoveWithCam;
    bool nvidia;
    osg::Program* pgm1;
    osg::Group* kinectgrp;
    float initialPointScale;// = ConfigManager::getFloat("Plugin.Points.PointScale", 0.001f);



    bool kLockRot;
    bool kLockScale;
    bool kLockPos;
    bool kShowArtifactPanel;
    bool kShowInfoPanel;
    std::vector<SelectableItem> selectableItems;
    void kinectInit();
    void kinectOff();
    void moveCam(double, double, double, double, double, double, double, double);
    void createSelObj(osg::Vec3 pos, std::string, float radius);
    void ThirdInit();
    void ThirdLoop();
    double depth_to_hue(double dmin, double depth, double dmax);
    void HSVtoRGB(float* r, float* g, float* b, float h, float s, float v);
    bool handApproachingDisplayPerimeter(float x, float y, int roi);
    void cloudOff();
    void navOff();
    void navOn();
    void cloudOn();
    void colorOff();
    void colorOn();
    void gesturesOff();
    void gesturesOn();
    void moveWithCamOff();
    void moveWithCamOn();
    void updateInfoPanel();

    osg::ref_ptr<osg::Vec4dArray> kinectColours;
    osg::ref_ptr<osg::Vec3Array> kinectVertices;
    ///Kinect
protected:

    static ArtifactVis2* _artifactvis2;
    int _testA;

    struct Locus
    {
        std::vector<std::string> fields;
        std::vector<std::string> values;
        std::string id;
        std::string name;
        osg::Geometry* geom;
        osg::Geometry* tgeom;
        osg::ref_ptr<osg::Geode> fill_geode;
        osg::ref_ptr<osg::Geode> line_geode;
        osg::ref_ptr<osg::Geode > top_geode;
        osg::Geode* text_geode;
        bool visible;
        std::vector<osg::Vec3d> coordsTop;
        std::vector<osg::Vec3d> coordsBot;
        osgText::Text* label;
    };
    struct QueryGroup
    {
        std::string name;
        std::string query;
        bool sf;
        std::vector<Artifact*> artifacts;
        std::vector<Locus*> loci;
        bool active;
        std::string timestamp;
        std::string kmlPath;
        osg::ref_ptr<osg::MatrixTransform> sphereRoot;
        bool updated;
        osg::Vec3d center;
    };
    struct Table
    {
        std::string name;
        cvr::SubMenu* queryMenu;
        cvr::MenuButton* genQuery;
        cvr::MenuButton* clearConditions;
        cvr::MenuButton* saveQuery;
        cvr::SubMenu* viewQuery;
        cvr::MenuButton* addOR;
        cvr::MenuButton* removeOR;
        cvr::SubMenu* conditions;
        std::vector<cvr::SubMenu*> querySubMenu;
        std::vector<cvr::SubMenu*> querySubMenuSlider;
        std::vector<cvr::MenuTextButtonSet*> queryOptions;
        std::vector<cvr::MenuCheckbox*> querySlider;
        std::vector<std::vector<std::string> > sliderEntry;
        std::vector<cvr::MenuList*> queryOptionsSlider;
        cvr::MenuText* query_view;
        std::string current_query;
    };
    struct FlyPlace
    {
        std::vector<std::string> name;
        std::vector<float> scale;
        std::vector<double> x;
        std::vector<double> y;
        std::vector<double> z;
        std::vector<double> rx;
        std::vector<double> ry;
        std::vector<double> rz;
        std::vector<double> rw;

    };
    FlyPlace* _flyplace;
    osg::ref_ptr<osg::Node> _models[676];
    bool _modelLoaded[676];
    cvr::SubMenu* _modelDisplayMenu;
    std::vector<cvr::MenuCheckbox*> _showModelCB;
    std::vector<cvr::MenuButton*> _reloadModel;
    cvr::SubMenu* _pcDisplayMenu;
    std::vector<cvr::MenuCheckbox*> _showPCCB;
    std::vector<cvr::MenuButton*> _reloadPC;
    cvr::SubMenu* _avMenu;
    cvr::SubMenu* _displayMenu;
    cvr::SubMenu* _artifactDisplayMenu;
    std::vector<cvr::SubMenu*> _queryOptionMenu;
    std::vector<cvr::MenuCheckbox*> _queryOption;
    std::vector<cvr::SubMenu*> _showQueryInfo;
    std::vector<cvr::MenuCheckbox*> _queryDynamicUpdate;
    std::vector<cvr::MenuText*> _queryInfo;
    std::vector<cvr::MenuButton*> _eraseQuery;
    std::vector<cvr::MenuButton*> _centerQuery;
    std::vector<cvr::MenuCheckbox*> _toggleLabel;  //new
    cvr::SubMenu* _locusDisplayMenu;
    cvr::MenuTextButtonSet* _locusDisplayMode;
    cvr::MenuCheckbox* _selectArtifactCB;
    cvr::MenuCheckbox* _manipArtifactCB;
    //Kinect
    cvr::MenuCheckbox* _selectKinectCB;
    cvr::MenuCheckbox* _kColorOn;
    cvr::MenuCheckbox* _kShowColor;
    cvr::MenuCheckbox* _kShowPCloud;
    cvr::MenuCheckbox* _kMoveWithCam;
    cvr::MenuCheckbox* _scaleBar;
    cvr::MenuCheckbox* _kinectOn;
    cvr::MenuRangeValue* _kColorFPS;
    cvr::MenuCheckbox* _kNavSpheres;
    cvr::MenuCheckbox* _kUseGestures;
    cvr::MenuCheckbox* _kShowArtifactPanel;
    cvr::MenuCheckbox* _kShowInfoPanel;
    cvr::SubMenu* _kinectMenu;

    cvr::MenuCheckbox* _kLockRot;
    cvr::MenuCheckbox* _kLockScale;
    cvr::MenuCheckbox* _kLockPos;

    float distanceMIN, distanceMAX;

    ///Kinect
    cvr::SubMenu* _flyMenu;
    std::vector<cvr::MenuButton*> _goto;
    cvr::MenuButton* _bookmarkLoc;
    //cvr::MenuButton * _goto; //new
    cvr::MenuCheckbox* _selectCB;
    cvr::SubMenu* _tablesMenu;
    osg::MatrixTransform* _root;

    cvr::TabbedDialogPanel* _artifactPanel;
    cvr::TabbedDialogPanel* _infoPanel;
    cvr::DialogPanel* _selectionStatsPanel;

    std::string _picFolder;

    std::vector<std::string> _dcList;
    std::map<std::string, bool> _dcVisibleMap;

    std::vector<Table*> _tables;
    std::vector<QueryGroup*> _query;

    osg::Vec3 _selectStart;
    osg::Vec3 _selectCurrent;

    osg::Material* _defaultMaterial;

    osg::Vec4 _colors[729];

    //osg::LOD * _my_own_root;
    std::vector<osg::Vec3Array* > _coordsPC;
    std::vector<osg::Vec4Array* > _colorsPC;
    std::vector<osg::Vec3d> _avgOffset;
    std::vector<osg::Node* > _modelSFileNode;
    std::vector<osg::ref_ptr<osg::MatrixTransform> > _siteRoot;
    std::vector<osg::ref_ptr<osg::MatrixTransform> > _pcRoot;
    std::vector<osg::Vec3d> _sitePos;
    std::vector<osg::Vec3d> _siteScale;
    std::vector<osg::Vec3d> _siteRot;
    std::vector<osg::Vec3d> _pcPos;
    std::vector<osg::Vec3d> _pcScale;
    std::vector<osg::Vec3d> _pcRot;
    std::vector<int> _pcFactor; //New
    osg::ref_ptr<osg::MatrixTransform> _selectBox;
    osg::ref_ptr<osg::MatrixTransform> _selectMark;
    osg::ref_ptr<osg::MatrixTransform> _selectModelLoad; //New Add

    osg::Node* _modelFileNode;
    osg::Node* _modelFileNode2;

    double _selectRotx;
    double _selectRoty;
    double _selectRotz;
    double _snum;
    osg::ref_ptr<osg::MatrixTransform> _scaleBarModel; //New Add
    osg::Vec3 _modelartPos; //New Add
    double _xRotMouse; //New Add
    double _yRotMouse; //New Add
    //float _LODmaxRange;
    float _sphereRadius;


    bool _ossim;
    bool _osgearth;

    std::string getTimeModified(std::string file);
    int dc2Int(std::string dc);
    void loadScaleBar(osg::Vec3d start);  //new
    void setActiveArtifact(int _lockedTo, int _lockedType, int art, int query);
    void readQuery(QueryGroup* query);
    std::vector<std::string> getSelectedQueries();
    void listArtifacts();
    void displayArtifacts(QueryGroup* query);
    osg::Drawable* createObject(std::string dc, float tessellation, osg::Vec3d& pos);
    void clearConditions(Table* t);
    void readPointCloud(int index);
    void readSiteFile(int index);
    void readLocusFile(QueryGroup* query);
    void setupSiteMenu();
    void reloadSite(int index);  //New
    void setupLocusMenu();
    void setupQuerySelectMenu();
    void setupTablesMenu();
    void setupQueryMenu(Table* table);
    void updateSelect();
    std::string getCurrentQuery(Table* t);
    bool modelExists(const char* filename);
    void loadModels();
    void rotateModel(double rx, double ry, double rz);
    void setupFlyToMenu();

    //Space Navigator
    float transMult, rotMult;
    float transcale, rotscale;

};

#endif
