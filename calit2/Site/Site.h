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

#include <kernel/CVRPlugin.h>
#include <kernel/FileHandler.h>
#include <kernel/ScreenBase.h>
#include <kernel/SceneManager.h>
#include <kernel/Navigation.h>
#include <kernel/ComController.h>
#include <config/ConfigManager.h>


const int selection = 2;

using namespace std;

class Site : public cvr::CVRPlugin, public cvr::FileLoadCallback
{
  private:
    bool joyStickReset;
    int currentMode;
    vector< osg::ref_ptr<osg::Texture2D> > texFiles;
    osg::StateSet* texstate;
    int numberDivisions;
    int totalNumPoints;
    int level;
    int textureVisible; 
    osg::Geometry* radar;
    int dimensions[2];
    osg::Uniform* pixelsize;
    osg::Uniform* density;
    osg::Group* group;
    void createSite();
    void loadPointData(char *, osg::Group *);

  public:
    Site();
    virtual ~Site();
    bool init();
    int unloadFile();
    int loadFile(const char*);
    void preFrame();
    virtual bool loadFile(string filename);
};
#endif
