#ifndef _POINTS_H
#define _POINTS_H

#include <queue>
#include <vector>

// CVR
#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/ScreenBase.h>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/Navigation.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrMenu/MenuSystem.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/FileHandler.h>


// OSG
#include <osg/Group>
#include <osg/Vec3>
#include <osg/Uniform>
#include <osgDB/ReadFile>
#include <osgDB/FileUtils>

class Points : public cvr::CVRPlugin, public cvr::FileLoadCallback
{
  private:
    osg::Group* group;
    osg::Uniform* objectScale;
    osg::Uniform* pointScale;
    osg::Program* pgm1;
    void readXYZ(std::string& filename, osg::Vec3Array* points, osg::Vec4Array* colors);
    void readXYB(std::string& filename, osg::Vec3Array* points, osg::Vec4Array* colors);

  public:
    Points();
    virtual ~Points();
    bool init();
    virtual bool loadFile(std::string file);
    bool loadFile(std::string file, osg::Group * grp);
    void preFrame();
    void message(int type, char *&data, bool collaborative=false);
};
#endif
