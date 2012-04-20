#ifndef _LOCATIONTRACKER_
#define _LOCATIONTRACKER_

#include <cvrKernel/CVRPlugin.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuButton.h>

#include <osgDB/FileUtils>
#include <osgDB/FileNameUtils>

#include <osg/MatrixTransform>

#include <string>
#include <vector>

#define MYSQLPP_MYSQL_HEADERS_BURIED
#include <mysql/mysql.h>
#include <mysql++/mysql++.h>

class LocationTracker : public cvr::CVRPlugin
{
    public:        
        LocationTracker();
        virtual ~LocationTracker();
        
	bool init();
        void message(int type, char * data);

    protected:
};

#endif
