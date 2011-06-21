#ifndef _OSSIM_PLANET_H
#define _OSSIM_PLANET_H

#include <ossim/base/ossimPreferences.h>
#include <ossim/init/ossimInit.h>
#include <ossim/base/ossimFilename.h>
#include <ossim/base/ossimEnvironmentUtility.h>
#include <ossim/base/ossimDirectory.h>
#include <ossim/base/ossimKeywordlist.h>
#include <ossim/base/ossimKeyword.h>
#include <ossim/init/ossimInit.h>
#include <ossim/elevation/ossimElevManager.h>
#include <ossim/base/ossimApplicationUsage.h>
#include <ossim/projection/ossimUtmpt.h>
#include <ossim/base/ossimGpt.h>
#include <osgViewer/ViewerEventHandlers>
#include <ossimPlanet/ossimPlanetLatLonHud.h>
#include <ossimPlanet/ossimPlanetKmlLayer.h>
#include <ossimPlanet/ossimPlanetTerrain.h>
#include <ossimPlanet/ossimPlanetTerrainGeometryTechnique.h>
#include <ossimPlanet/ossimPlanetTextureLayerRegistry.h>
#include <ossimPlanet/ossimPlanetCloudLayer.h>
#include <ossimPlanet/ossimPlanet.h>
#include <ossimPlanet/ossimPlanetLand.h>
#include <ossimPlanet/ossimPlanetEphemeris.h>
#include <ossimPlanet/ossimPlanetCallback.h>
#include <ossimPlanet/ossimPlanetReentrantMutex.h>
#include <ossimPlanet/ossimPlanetLookAt.h>

// OSG:
#include <osg/Node>
#include <osg/Switch>
#include <osg/CullFace>
#include <osg/Sequence>
#include <osg/Geode>
#include <osg/LineWidth>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/PolygonMode>
#include <osg/NodeVisitor>
#include <osg/Vec3>
#include <osg/MatrixTransform>
#include <osg/Matrix>
#include <osg/LightSource>
#include <osgViewer/View>
#include <osgViewer/Viewer>
#include <osgUtil/SceneView>
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgText/Text>
#include <osgUtil/IntersectVisitor>
#include <osg/ShapeDrawable>
#include <kernel/CVRViewer.h>
#include <kernel/PluginHelper.h>

#include <kernel/CVRPlugin.h>
#include <kernel/ScreenBase.h>
#include <kernel/SceneManager.h>
#include <kernel/Navigation.h>
#include <kernel/ComController.h>
#include <config/ConfigManager.h>
#include <menu/SubMenu.h>
#include <menu/MenuCheckbox.h>

class OssimPlanet;

class ossimPlanetViewerCallback : public ossimPlanetCallback
{
public:
   virtual void viewChanged(osgViewer::Viewer* /*viewer*/){}
};


// post update for shifting matching camera for cloud cover
struct PostTraversal: public cvr::CVRViewer::UpdateTraversal
{
    private:
   	osg::Camera* theEphemerisCamera;
        osg::Matrix planetRescale;

    public:
        PostTraversal(osg::Camera*, osg::Matrix rescale);
        void update();
};


// used to pre compute the current location of the camera relative to the planet in lat lon and height
struct PreTraversal: public cvr::CVRViewer::UpdateTraversal
{
    private:
	osg::Camera* defaultCamera;
	osg::Matrix theCurrentViewMatrix;
	osg::Matrix theCurrentViewMatrixInverse;
	OssimPlanet* planetplugin;
	ossimPlanetLookAt* theCurrentLookAt;
	ossimPlanetLookAt* theCurrentCamera;
        osg::Matrix planetRescale;

    public:
	PreTraversal(osg::Camera*, OssimPlanet*, ossimPlanetLookAt* look, ossimPlanetLookAt* camera, osg::Matrix rescale);
	void update();
};

class OssimPlanet : public cvr::CVRPlugin, public cvr::MenuCallback, public ossimPlanetCallback, public ossimPlanetCallbackListInterface<ossimPlanetViewerCallback>
{
  private:
    static ossimPlanet* planet;
    static OssimPlanet* oplanet;
    void cloudCover();
    ossimPlanetEphemeris* ephemeris;
    void getObjectIntersection(osg::Node *root, osg::Vec3& wPointerStart, osg::Vec3& wPointerEnd, IsectInfo& isect);
    void processNav(double speed);
    void processMouseNav(double speed);
    double getSpeed(double distance);

    cvr::SubMenu * _ossimMenu;
    cvr::MenuCheckbox * _navCB;

    bool _navActive;
    int _navHand;
    osg::Matrix _navHandMat;

    bool _mouseNavActive;
    int _startX,_startY;
    int _currentX,_currentY;
    bool _movePointValid;
    osg::Vec3d _movePoint;
    //osg::MatrixTransform * _testMark;

  public:
    OssimPlanet();
    virtual ~OssimPlanet();
    // allow addition of model to planet
    bool addModel(osg::Node*, double lat, double lon, osg::Vec3 scale = osg::Vec3(1.0, 1.0, 1.0), double height = 0.0, double heading = 0.0, double pitch = 0.0, double roll = 0.0); // height about ground level
    osg::Vec3d convertUtm84ToLatLon(int zone, char hemi, double northing, double easting); 
    bool init();
    void preFrame();
    int getPriority() { return 30; }
    bool buttonEvent(int type, int button, int hand, const osg::Matrix & mat);
    bool mouseButtonEvent (int type, int button, int x, int y, const osg::Matrix &mat);
    void menuCallback(cvr::MenuItem * item);

    static OssimPlanet* instance();
};
#endif
