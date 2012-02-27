#ifndef _OSGEARTH_
#define _OSGEARTH_

#include <kernel/CVRPlugin.h>
#include <kernel/InteractionManager.h>
#include <menu/SubMenu.h>
#include <menu/MenuButton.h>
#include <menu/MenuCheckbox.h>

#include <osgEarth/Map>
#include <osgEarth/MapNode>
#include <osgEarthUtil/EarthManipulator>
#include <osgEarth/Utils>
#include <osgEarth/CompositeTileSource>

#include <osgDB/FileUtils>
#include <osgDB/FileNameUtils>

#include <osgEarthDrivers/gdal/GDALOptions>
#include <osgEarthDrivers/tms/TMSOptions>
#include <osgEarthDrivers/arcgis/ArcGISOptions>

#include <osg/MatrixTransform>

#include <string>
#include <vector>

class OsgEarth : public cvr::CVRPlugin, public cvr::MenuCallback
{
    public:        
        OsgEarth();
        virtual ~OsgEarth();
        
	bool init();
        void message(int type, char * data);
        int getPriority() { return 51; }
	void preFrame();
        bool processEvent(cvr::InteractionEvent * event);
        bool buttonEvent(int type, int button, int hand, const osg::Matrix & mat);
        bool mouseButtonEvent (int type, int button, int x, int y, const osg::Matrix &mat);
        void menuCallback(cvr::MenuItem * item);
        double getSpeed(double distance);
        void processNav(double speed);
        void processMouseNav(double speed);


    protected:
        osgEarth::Map * map;

	cvr::SubMenu * _osgEarthMenu;
        cvr::MenuCheckbox * _navCB;

        bool _navActive;
        int _navHand;
        osg::Matrix _navHandMat;

        bool _mouseNavActive;
        int _startX,_startY;
        int _currentX,_currentY;
        bool _movePointValid;
        osg::Vec3d _movePoint;
                
};

#endif
