#ifndef _SAMPLEEARTH_
#define _SAMPLEEARTH_

#include <cvrKernel/CVRPlugin.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuButton.h>

#include <osgEarth/Map>
#include <osgEarth/MapNode>
#include <osgEarth/Utils>

#include <osgDB/FileUtils>
#include <osgDB/FileNameUtils>

#include <osg/MatrixTransform>

#include <string>
#include <vector>

class SampleEarth : public cvr::CVRPlugin
{
    public:        
        SampleEarth();
        virtual ~SampleEarth();
        
	bool init();

    protected:
        osgEarth::Map * map;
                
};

#endif
