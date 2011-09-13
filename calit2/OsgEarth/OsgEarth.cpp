#include "OsgEarth.h"

#include <config/ConfigManager.h>
#include <kernel/SceneManager.h>
#include <kernel/PluginManager.h>
#include <menu/MenuSystem.h>
#include <PluginMessageType.h>
#include <iostream>

#include <osg/Matrix>
#include <osgDB/ReadFile>
#include <osg/Geode>
#include <osg/ShapeDrawable>

using namespace osg;
using namespace std;
using namespace cvr;
using namespace osgEarth;
using namespace osgEarth::Drivers;
using namespace osgEarth::Util;


CVRPLUGIN(OsgEarth)

OsgEarth::OsgEarth()
{

}

void OsgEarth::message(int type, char * data)
{
    // data needs to include the plugin name and also the lat,lon and height
    if(type == OE_ADD_MODEL)
    {
	// data contains 3 floats
	OsgEarthRequest request = * (OsgEarthRequest*) data;

	// if get a request create new node add matrix and return the address of the matrixtransform
        osg::Matrixd output;
	map->getProfile()->getSRS()->getEllipsoid()->computeLocalToWorldTransformFromLatLongHeight(
	        osg::DegreesToRadians(request.lat),
	        osg::DegreesToRadians(request.lon),
		request.height,
	        output );

	if( request.trans != NULL )
	{
            request.trans->setMatrix(output);
	}
	else
	{
	    osg::MatrixTransform* trans = new osg::MatrixTransform();
            trans->setMatrix(output);
            SceneManager::instance()->getObjectsRoot()->addChild(trans);
            request.trans = trans;
	}

        // send message back	
	PluginManager::instance()->sendMessageByName(request.pluginName,OE_TRANSFORM_POINTER,(char *) &request);
    }
}

bool OsgEarth::init()
{
    std::cerr << "OsgEarth init\n";

    // Start by creating the map:
    map = new Map();

    //Add a base layer
    //GDALOptions basemapOpt;
    //basemapOpt.url() = "/home/covise/data/BMNG/world.topo.bathy.200401.3x86400x43200.tif";
    //map->addImageLayer( new ImageLayer( ImageLayerOptions("basemap", basemapOpt) ) );

    //Add a base layer
    //TMSOptions bluemarbleOpt;
    //bluemarbleOpt.url() = "http://demo.pelicanmapping.com/rmweb/data/bluemarble-tms/tms.xml";
    //map->addImageLayer( new ImageLayer( ImageLayerOptions("bluemarble", bluemarbleOpt) ) );

    //Add a base layer
    ArcGISOptions mapserverOpt;
    mapserverOpt.url() = "http://server.arcgisonline.com/ArcGIS/rest/services/ESRI_Imagery_World_2D/MapServer";
    map->addImageLayer( new ImageLayer( ImageLayerOptions("mapserver", mapserverOpt) ) );

    // add a TMS fire layer:
    //TMSOptions firetms;
    //firetms.url() = "http://lava.ucsd.edu/FireTMS/tilemapresource.xml";
    //map->addImageLayer( new ImageLayer( ImageLayerOptions( "firetms", firetms ) ) );

    // add a TMS elevation layer:
    TMSOptions elevationtms;
    elevationtms.url() = "http://demo.pelicanmapping.com/rmweb/data/srtm30_plus_tms/tms.xml";
    map->addElevationLayer( new ElevationLayer( "SRTM", elevationtms ) );

    // add a local elevation layer (san diego)
    //GDALOptions elevation;
    //elevation.url() = "/home/covise/data/elevation/sdmrg_elev.tif";
    //map->addElevationLayer( new ElevationLayer("Elevation", elevation) );

    //http://hyperquad.ucsd.edu/cgi-bin/irene?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetCapabilities
    //add san diego fire imagery
    //GDALOptions sandiegoOpt;
    //sandiegoOpt.url() = "/state/partition1/data/SanDiego/bigMergeSanDiego2.tif";
    //map->addImageLayer( new ImageLayer( ImageLayerOptions("sanDiegoFire", sandiegoOpt) ) );

/*
    // lat, lon, height (meters, about ground level??)
    osg::Matrixd output;
    map->getProfile()->getSRS()->getEllipsoid()->computeLocalToWorldTransformFromLatLongHeight(
                osg::DegreesToRadians(32.73f),
		osg::DegreesToRadians(-117.17f),
		0.0f,
		output );

    printf("Trans is %f %f %f\n", output.getTrans().x(), output.getTrans().y(), output.getTrans().z());

    // creating a rectangular box to add to the surface for testing ( orientation and size )
    osg::Geode* geode = new osg::Geode();
    geode->addDrawable(new osg::ShapeDrawable(new osg::Box(osg::Vec3(0.0f,0.0f,0.0f),2000.0, 3000.0, 4000.0)));
    osg::MatrixTransform* trans = new osg::MatrixTransform();
    trans->setMatrix(output);
    trans->addChild(geode);
    SceneManager::instance()->getObjectsRoot()->addChild(trans);
*/
/*
    osgEarth::CompositeTileSourceOptions compositeOpt;
    for (unsigned int i = 0; i < files.size(); i++)
    {
       GDALOptions gdalOpt;
       gdalOpt.url() = files[i];
       ImageLayerOptions ilo(files[i], gdalOpt);
       //Set the transparent color on each image        
       ilo.transparentColor() = osg::Vec4ub(255, 255, 206, 0); 
       compositeOpt.add( ilo );
    }

    map->addImageLayer( new ImageLayer( ImageLayerOptions("composite", compositeOpt) ) );
*/
    MapNodeOptions mapNodeOptions;
    mapNodeOptions.enableLighting() = false;
    
    MapNode* mapNode = new MapNode( map , mapNodeOptions);

/*
    //add sky
    double hours = skyConf.value( "hours", 12.0 );
    SkyNode* s_sky = new SkyNode( mapNode->getMap() );
    s_sky->setDateTime( 2011, 3, 6, hours );
    s_sky->attach( &viewer ); // NEED TO GET ACCESS TO THE VIEWER
    SceneManager::instance()->getObjectsRoot()->addChild( s_sky );
*/

    SceneManager::instance()->getObjectsRoot()->addChild(mapNode);

    return true;
}


OsgEarth::~OsgEarth()
{
}
