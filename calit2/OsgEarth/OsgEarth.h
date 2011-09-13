#ifndef _OSGEARTH_
#define _OSGEARTH_

#include <kernel/CVRPlugin.h>
#include <menu/SubMenu.h>
#include <menu/MenuButton.h>

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

class OsgEarth : public cvr::CVRPlugin
{
    public:        
        OsgEarth();
        virtual ~OsgEarth();
        
	bool init();
        void message(int type, char * data);
        int getPriority() { return 51; }

    protected:
        osgEarth::Map * map;
                
};

#endif
