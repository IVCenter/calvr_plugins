#ifndef _SITE_H
#define _SITE_H

#include <queue>
#include <vector>

#include <osgTerrain/TerrainTile>
#include <osgTerrain/GeometryTechnique>
#include <osgTerrain/Layer>

// OSG

#include <osg/Group>
#include <osg/Sequence>
#include <osg/Vec3>

#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/FileHandler.h>
#include <cvrKernel/Screens/ScreenBase.h>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/Navigation.h>
#include <cvrKernel/ComController.h>
#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrMenu/MenuSystem.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuRangeValue.h>



const int selection = 2;

using namespace std;

class Site : public cvr::CVRPlugin, public cvr::MenuCallback, public cvr::FileLoadCallback
{
  private:
    int currentMode;
    vector< osg::ref_ptr<osg::Texture2D> > texFiles;
    osg::StateSet* texstate;
    int numberDivisions;
    int totalNumPoints;
    osg::Geometry* radar;
    int dimensions[2];
    osg::Uniform* objectScale;
    osg::Uniform* pointSize;
    osg::Group* group;
    void createSite();
    void loadPointData(string, osg::Group *);

    //menu items
    cvr::SubMenu* siteSubMenuItem;
    cvr::SubMenu* textureSubMenuItem;
    cvr::MenuRangeValue * rangeMenuItem;

  public:
    Site();
    virtual ~Site();
    bool init();
    void preFrame();
    virtual bool loadFile(string filename);
    void menuCallback(cvr::MenuItem * item);
};
#endif
