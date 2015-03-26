#ifndef _POINTSOOC_H
#define _POINTSOOC_H

#include <queue>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>

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

class PointsOOC : public cvr::CVRPlugin, public cvr::MenuCallback, public cvr::FileLoadCallback
{
  protected:

	// persistant data to save for point data
	struct Loc
	{
		osg::Matrix pos;
		float pointSize;
		float maxSize;
        float pointFunc[3];
		float shaderSize;
		float pointAlpha;
		float shaderEnabled;
        osg::ref_ptr<osg::StateSet> shaderStateset;
        osg::ref_ptr<osg::StateSet> pointStateset;
	};
 
    void writeConfigFile();
	void adjustLODScale(float lower = 20.0, float upper = 50.0, float interval = 0.25);
    void removeAll();

    // context map
    std::map<cvr::SceneObject*,cvr::MenuRangeValue*> _sliderPointSizeMap;
    std::map<cvr::SceneObject*,cvr::MenuRangeValue*> _sliderShaderSizeMap;
    std::map<cvr::SceneObject*,cvr::MenuRangeValue*> _sliderPointFuncMap; // size func
	std::map<cvr::SceneObject*,cvr::MenuRangeValue*> _sliderAlphaMap;
    std::map<cvr::SceneObject*,cvr::MenuButton*> _deleteMap;
    std::map<cvr::SceneObject*,cvr::MenuButton*> _saveMap;
    std::map<cvr::SceneObject*,cvr::MenuCheckbox*> _boundsMap;
    std::map<cvr::SceneObject*,cvr::MenuCheckbox*> _shaderMap;
    std::map<cvr::SceneObject*,cvr::SubMenu*> _subMenuMap;
    std::vector<cvr::SceneObject*> _loadedPoints;

    std::map<std::string, Loc > _locInit;
    std::string _configPath;

    cvr::SubMenu * _mainMenu, * _loadMenu;
    cvr::MenuButton * _removeButton;
    std::vector<cvr::MenuButton*> _menuFileList;
    std::vector<std::string> _filePaths;

	float _startTime;
	unsigned int _currentFrame;

  public:
    PointsOOC();
    virtual ~PointsOOC();
    bool init();
    virtual bool loadFile(std::string file);
    void menuCallback(cvr::MenuItem * item);
    void preFrame();
    void message(int type, char *&data, bool collaborative=false);
};
#endif
