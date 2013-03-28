#ifndef _POINTS_H
#define _POINTS_H

#include <queue>
#include <vector>

// CVR
#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/ScreenBase.h>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/SceneObject.h>
#include <cvrKernel/Navigation.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/ComController.h>
#include <cvrMenu/MenuSystem.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuRangeValue.h>
#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/FileHandler.h>

// OSG
#include <osg/Group>
#include <osg/Vec3>
#include <osg/Uniform>
#include <osgDB/ReadFile>
#include <osgDB/FileUtils>

class Points : public cvr::CVRPlugin, public cvr::MenuCallback, public cvr::FileLoadCallback
{
  protected:
    osg::ref_ptr<osg::Program> pgm1;
    float initialPointScale;
 
    // container to hold pdf data
    struct PointObject
    {
       std::string name;
       cvr::SceneObject* scene;
       osg::Geode* points;
       osg::Uniform* pointScale;
       osg::Uniform* objectScale;
    }; 

    void readXYZ(std::string& filename, osg::Vec3Array* points, osg::Vec4Array* colors);
    void readXYB(std::string& filename, osg::Vec3Array* points, osg::Vec4Array* colors);
    void writeConfigFile();

    // context map
    std::map<struct PointObject*,cvr::MenuRangeValue*> _sliderMap;
    std::map<struct PointObject*,cvr::MenuButton*> _deleteMap;
    std::map<struct PointObject*,cvr::MenuButton*> _saveMap;
    std::map<struct PointObject*,cvr::MenuCheckbox*> _boundsMap;
    std::vector<struct PointObject*> _loadedPoints;
    
    std::map<std::string, std::pair<float, osg::Matrix> > _locInit;
    std::string _configPath;

    osg::Uniform* objectScale;

  public:
    Points();
    virtual ~Points();
    bool init();
    virtual bool loadFile(std::string file);
    bool loadFile(std::string file, osg::Group * grp);
    void menuCallback(cvr::MenuItem * item);
    void preFrame();
    void message(int type, char *&data, bool collaborative=false);
};
#endif
