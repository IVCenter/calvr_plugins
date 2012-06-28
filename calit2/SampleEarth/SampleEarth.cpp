#include "SampleEarth.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/PluginManager.h>
#include <cvrMenu/MenuSystem.h>
#include <iostream>

#include <osg/Matrix>
#include <osgDB/ReadFile>
#include <osg/Geode>
#include <osg/ShapeDrawable>

using namespace osg;
using namespace std;
using namespace cvr;
using namespace osgEarth;


CVRPLUGIN(SampleEarth)

SampleEarth::SampleEarth()
{

}

bool SampleEarth::init()
{
    std::cerr << "SampleEarth init\n";
    map = NULL;

    // try finding an exiusting planet attached to the scenegraph
    osgEarth::MapNode* mapNode = MapNode::findMapNode( SceneManager::instance()->getObjectsRoot() );
    
    // will return true if OsgEarth is enabled in config file
    if( mapNode )
    {
	printf("OsgEarth model was enabled in config file\n");
    	map = mapNode->getMap();

	// sample: compute a location on the planet
    	double lat = 32.73;  //degrees
    	double lon = -117.17; //degrees
    	double height = 0.0;   // on the surface (in meters)

    	osg::Matrixd output;
    	map->getProfile()->getSRS()->getEllipsoid()->computeLocalToWorldTransformFromLatLongHeight(
                osg::DegreesToRadians(lat),
                osg::DegreesToRadians(lon),
                height,
                output );

    	// attach a silly shape
    	osg::Geode* geode = new osg::Geode();
	osg::ShapeDrawable* shape = new osg::ShapeDrawable(new osg::Box(osg::Vec3(0.0, 0.0, -20000.0), 10000.0, 10000.0, 40000.0));
	geode->addDrawable(shape);

	osg::MatrixTransform * mat = new osg::MatrixTransform();
	mat->setMatrix(output);
	mat->addChild(geode);

    	SceneManager::instance()->getObjectsRoot()->addChild(mat);

    }
    else // failed force OsgEarth to load
    {
	printf("Enable OsgPlugin in configuration file\n");	
    }

    return true;
}


SampleEarth::~SampleEarth()
{
}
