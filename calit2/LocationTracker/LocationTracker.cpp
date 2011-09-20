#include "LocationTracker.h"

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

#include <iostream>

using namespace osg;
using namespace std;
using namespace cvr;


CVRPLUGIN(LocationTracker)

LocationTracker::LocationTracker()
{

}

void LocationTracker::message(int type, char * data)
{
    // data needs to include the plugin name and also the lat,lon and height
    if(type == OE_TRANSFORM_POINTER)
    {

       OsgEarthRequest * request = (OsgEarthRequest*) data;
    
       osg::Geode* geode = new osg::Geode();
       geode->addDrawable(new osg::ShapeDrawable(new osg::Box(osg::Vec3(0.0f,0.0f,0.0f),2000.0, 3000.0, 4000.0)));
       osg::MatrixTransform * trans = new osg::MatrixTransform();
       request->trans->addChild(geode);
    }
}

bool LocationTracker::init()
{
    std::cerr << "LocationTracker init\n";
    
    char * pluginName = "LocationTracker";


    // create database connection and request tour info
    mysqlpp::Connection*  conn = new mysqlpp::Connection(false);
    if(!conn->connect("test", "android.calit2.net", "pweber", "kacsttour"))
    {
         cerr << "Unable to connect to Android DB." << endl;
         delete conn;
         conn = NULL;
    }
    else
    {
         cerr << "Connected to Android DB." << endl;

	 stringstream querys;
         querys << "select filename, add_date from media where SnapshotID = 155;";
	 mysqlpp::Query query = conn->query(querys.str().c_str());
	 mysqlpp::StoreQueryResult res = query.store();
         for(int i = 0; i < res.num_rows(); i++)
         {
               int date = atoi(res[i]["add_date"].c_str());
               std::string filename = string(res[i]["filename"]);

	       printf("Date %d and file name is %s\n", date, filename.c_str());
         }

    }

/*
    OsgEarthRequest request;
    request.lat = 32.73f;
    request.lon = -117.17f;
    request.height = 2000.0f;
    request.trans = NULL;
    strcpy(request.pluginName, pluginName);

    // send message back	
    PluginManager::instance()->sendMessageByName("OsgEarth",OE_ADD_MODEL,(char *) &request);
*/
    return true;
}


LocationTracker::~LocationTracker()
{
}
