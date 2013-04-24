#ifndef WATERMAZE_PLUGIN_H
#define WATERMAZE_PLUGIN_H


#include <cvrKernel/ComController.h>
#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/FileHandler.h>
#include <cvrKernel/InteractionManager.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/SceneObject.h>

#include <cvrUtil/Intersection.h>
#include <cvrConfig/ConfigManager.h>

#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuRangeValue.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuText.h>

#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/MatrixTransform>
#include <osg/PositionAttitudeTransform>
#include <osg/ShapeDrawable>
#include <osg/Switch>
#include <osg/Texture2D>
#include <osgText/Text>
#include <osgDB/ReadFile>

#include <string.h>
#include <vector>
#include <map>

#include <iostream>
#include <stdio.h>
#include <netdb.h>
#include <sys/socket.h>
#include <X11/Xlib.h>
#include <spnav.h>

//#include <ftdi.h>
#include <ftd2xx.h>

namespace WaterMaze
{

class WaterMaze: public cvr::CVRPlugin, public cvr::MenuCallback
{
    public:
        WaterMaze();
        virtual ~WaterMaze();
        static WaterMaze * instance();
        bool init();
        void menuCallback(cvr::MenuItem * item);
        void preFrame();
        bool processEvent(cvr::InteractionEvent * event);
        void load();

    protected:
        void loadModels();
        void clear();
        void reset();
        void newHiddenTile();

        int init_SPP(int port); 
        void close_SPP();
        void write_SPP(int bytes, unsigned char* buf);
        void connectToServer();

        float randomFloat(float min, float max)
        {
            if (max < min) return 0;

            float random = ((float) rand()) / (float) RAND_MAX;
            float diff = max - min;
            float r = random * diff;
            return min + r;
        };

        static WaterMaze * _myPtr;

        cvr::SubMenu * _WaterMazeMenu;
        cvr::MenuButton * _loadButton, * _clearButton, *_newTileButton;
        cvr::MenuCheckbox * _gridCB;

        osg::ref_ptr<osg::MatrixTransform> _geoRoot; // root of all non-GUI plugin geometry
        osg::ref_ptr<osg::Switch> _gridSwitch; // grid on floor
        std::map<osg::Vec3, osg::Switch *> _tileSwitches;


        float widthTile, heightTile, numWidth, numHeight, depth, wallHeight,
            gridWidth;
        int _hiddenTile;
       
        bool _debug, _loaded; // turns on debug messages to command line

        // USB to Serial communication
        HANDLE hSerial;
        FT_HANDLE ftHandle;
        FT_STATUS ftStatus;
        DWORD devIndex;
        DWORD bytesWritten;
        unsigned char buf[16];
        bool _sppConnected;
};

};

#endif

