#ifndef _OSGEARTH_
#define _OSGEARTH_

#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/InteractionManager.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuCheckbox.h>

#include <osgEarth/Map>
#include <osgEarth/MapNode>
#include <osgEarth/Utils>
#include <osgEarth/TerrainOptions>

#include <osgDB/FileUtils>
#include <osgDB/FileNameUtils>

#include <osgEarthDrivers/tms/TMSOptions>
#include <osgEarthDrivers/arcgis/ArcGISOptions>
#include <osgEarthDrivers/gdal/GDALOptions>

#include <osg/MatrixTransform>

#include <string>
#include <vector>

class OsgEarth : public cvr::CVRPlugin, public cvr::MenuCallback
{
    public:        
        OsgEarth();
        virtual ~OsgEarth();
        
	bool init();
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
        osgEarth::Map * _map;
        osgEarth::MapNode * _mapNode;

        cvr::SubMenu * _osgEarthMenu;
        cvr::MenuCheckbox * _navCB;
        cvr::MenuCheckbox * _visCB;
            
        bool _navActive;
        int _navHand;
        osg::Matrix _navHandMat;
                       
        bool _mouseNavActive;
        int _startX,_startY;
        int _currentX,_currentY;
        bool _movePointValid;
        osg::Vec3d _movePoint;

 	//bool _setting_viewpoint;
	//osgEarthUtil::Viewpoint _pending_viewpoint;
        //double _pending_viewpoint_duration_s;
        //bool _has_pending_viewpoint;

	//void setViewPoint( const Viewpoint& vp, double duration_s = 0.0);
	//void cancelViewpointTransition() { _setting_viewpoint = false; }
};

#endif
