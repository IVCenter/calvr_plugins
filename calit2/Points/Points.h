#ifndef _POINTS_H
#define _POINTS_H

#include <queue>
#include <vector>

// CVR
#include <kernel/CVRPlugin.h>
#include <kernel/ScreenBase.h>
#include <kernel/SceneManager.h>
#include <kernel/Navigation.h>
#include <kernel/PluginHelper.h>
#include <menu/MenuSystem.h>
#include <menu/SubMenu.h>
#include <menu/MenuButton.h>
#include <menu/MenuCheckbox.h>
#include <config/ConfigManager.h>
#include <kernel/FileHandler.h>


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
    void preFrame();
};
#endif
