#include "OsgVnc.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/ComController.h>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/PluginManager.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/NodeMask.h>
#include <cvrKernel/InteractionManager.h>
#include <cvrMenu/MenuSystem.h>
#include <PluginMessageType.h>
#include <iostream>

#include <osg/Matrix>
#include <osgDB/ReadFile>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/StateSet>
#include <osg/Material>
#include <osg/Texture2D>
#include <osg/TextureRectangle>
#include <osg/TextureCubeMap>
#include <osg/TexMat>
#include <osg/CullFace>
#include <osg/ImageStream>
#include <osg/io_utils>
#include <osgDB/Registry>

#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>

using namespace osg;
using namespace std;
using namespace cvr;

// browser query commands
const std::string replHome = "repl.home();";
const std::string baseQuery = "content.location.href = 'http://www.google.com/search?as_q=";
const std::string replQuit = "repl.quit();";


CVRPLUGIN(OsgVnc)

OsgVnc::OsgVnc() : FileLoadCallback("vnc")
{
}

bool OsgVnc::loadFile(std::string filename)
{

    osgWidget::GeometryHints hints(osg::Vec3(0.0f,0.0f,0.0f),
                                   osg::Vec3(1.0f,0.0f,0.0f),
                                   osg::Vec3(0.0f,0.0f,1.0f),
                                   osg::Vec4(1.0f,1.0f,1.0f,1.0f),
                                   osgWidget::GeometryHints::RESIZE_HEIGHT_TO_MAINTAINCE_ASPECT_RATIO);

    
    std::string hostname = osgDB::getNameLessExtension(filename);
    osg::ref_ptr<osgWidget::VncClient> vncClient = new osgWidget::VncClient;
    if (vncClient->connect(hostname, hints))
    {
        struct VncObject * currentobject = new struct VncObject;
        currentobject->name = hostname;

        // add to scene object
        VncSceneObject * sot = new VncSceneObject(currentobject->name, vncClient.get(), ComController::instance()->isMaster(), false,false,false,true,true);
        PluginHelper::registerSceneObject(sot,"OsgVnc");

        // set up SceneObject
        sot->attachToScene();
        sot->setNavigationOn(true);
        sot->addMoveMenuItem();
        sot->addNavigationMenuItem();
        sot->addScaleMenuItem("Scale", 0.1, 10.0, 1.0);

        currentobject->scene = sot;
        
        // add controls
        //MenuCheckbox * mcb = new MenuCheckbox("Plane lock", true);
        //mcb->setCallback(this);
        //sot->addMenuItem(mcb);
        //_planeMap[currentobject] = mcb;
        
        MenuButton * mb = new MenuButton("Delete");
        mb->setCallback(this);
        sot->addMenuItem(mb);
        _deleteMap[currentobject] = mb;

        _loadedVncs.push_back(currentobject);
    }
    else
    {
	    std::cerr << "Unable to read vnc stream\n";
    }

    return true;
}

void OsgVnc::menuCallback(MenuItem* menuItem)
{

    //check map for a delete
    for(std::map<struct VncObject*, MenuButton*>::iterator it = _deleteMap.begin(); it != _deleteMap.end(); it++)
    {
        if(menuItem == it->second)
        {
            for(std::vector<struct VncObject*>::iterator delit = _loadedVncs.begin(); delit != _loadedVncs.end(); delit++)
            {
                if((*delit) == it->first)
                {
		            // need to delete title SceneObject
		            if( it->first->scene )
			            delete it->first->scene;
                    it->first->scene = NULL; 
    
                    _loadedVncs.erase(delit);
                    break;
                }
            }

            delete it->first;
            delete it->second;
            _deleteMap.erase(it);

            break;
        }
    }
}

bool OsgVnc::init()
{
    std::cerr << "OsgVnc init\n";
    return true;
}

void OsgVnc::message(int type, char *&data, bool collaborative)
{
    if(type == VNC_GOOGLE_QUERY)
    {
        if(collaborative)
        {
            return;
        }

		std::string hostname = ConfigManager::getEntry("Plugin.OsgVnc.BrowserQueryServer");
		int port = ConfigManager::getInt("Plugin.OsgVnc.Port", 4242);       

		// try and send request to remote firefox browser running mozrepl plugin
        OsgVncGoogleQueryRequest* request = (OsgVncGoogleQueryRequest*)data;        
		launchQuery(hostname, port, request->query);
    }
}

void OsgVnc::launchQuery(std::string& hostname, int portno, std::string& query)
{
	int sockfd, n;
    struct sockaddr_in serv_addr;
    struct hostent *server;

    sockfd = socket(AF_INET, SOCK_STREAM, 0);

    if (sockfd < 0)
	{
        std::cerr << "ERROR opening socket\n";
		return;	
	}

    server = gethostbyname(hostname.c_str());

    if (server == NULL) 
	{
        std::cerr << "ERROR, no such host\n";
        return;
    }

    bzero((char *) &serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    bcopy((char *)server->h_addr,
          (char *)&serv_addr.sin_addr.s_addr,
          server->h_length);
    serv_addr.sin_port = htons(portno);

    if (connect(sockfd,(struct sockaddr *) &serv_addr,sizeof(serv_addr)) < 0)
	{
        std::cerr << "ERROR connecting\n";
		return;
	}

    std::stringstream ss;
    ss << baseQuery;
    ss << query;
    ss << "';";

    // write initial debug message
    n = write(sockfd, replHome.c_str(), replHome.size());
    if (n < 0)
	{
        std::cerr << "ERROR writing to socket\n";
		return;
	}

    // read response
    char buffer[1024];
    bzero(buffer,1024);
    n = read(sockfd,buffer,1024);

	if(n < 0)
	{
        std::cerr << "ERROR reading from socket\n";
		return;
	}

    n = write(sockfd, ss.str().c_str(),ss.str().size());
    if (n < 0)
	{
        std::cerr << "ERROR writing to socket\n";
		return;
	}

    bzero(buffer,1024);
    n = read(sockfd,buffer,1024);

    n = write(sockfd, replQuit.c_str(), replQuit.size());
    if (n < 0)
	{
        std::cerr << "Error writing to socket\n";
		return;
	}

    bzero(buffer,1024);
    n = read(sockfd,buffer,1024);

    close(sockfd);
}

OsgVnc::~OsgVnc()
{

    while(_loadedVncs.size())
    {
         VncObject* it = _loadedVncs.at(0);
         
         // remove delete map item
         if(_deleteMap.find(it) != _deleteMap.end())
         {
            delete _deleteMap[it];
            _deleteMap.erase(it);
         }

		 if( it->scene )
		    delete it->scene;
         it->scene = NULL;

         _loadedVncs.erase(_loadedVncs.begin());
    }
}
