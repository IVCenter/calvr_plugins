#ifndef ELEVATOR_MODELHANDLER_H
#define ELEVATOR_MODELHANDLER_H 

#include <cvrKernel/PluginHelper.h>
#include <cvrConfig/ConfigManager.h>

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

#define NUM_DOORS 8
#define DOOR_SPEED 0.007
#define FLASH_SPEED 4
#define NUM_ALLY_FLASH 3
#define NUM_ALIEN_FLASH 8

namespace ElevatorRoom
{

enum Mode
{
    NONE,
    ALIEN,
    ALLY,
    CHECKER
};


class ModelHandler
{
    public:
        ModelHandler();
        ~ModelHandler();
        
        void update();
        void loadModels(osg::MatrixTransform* root);
        void openDoor();
        void closeDoor();

        void flashActiveLight();
        void flashCheckers();
        void flashAlien();
        void flashAlly();

        void setMode(Mode mode);
        void setActiveDoor(int doorNum);
        void setAlien(bool val);
        void setAlly(bool val);
        void setLight(bool val);

        osg::Geode * getActiveObject();
        float getDoorDistance();
        bool doorInView();

    protected:
        std::vector<osg::ref_ptr<osg::PositionAttitudeTransform> > _leftdoorPat, _rightdoorPat;
        std::vector<osg::ref_ptr<osg::ShapeDrawable> > _lights;

        // child 1 non-flashing, child 2 flashing
        std::vector<osg::ref_ptr<osg::Switch> > _aliensSwitch, _alliesSwitch, 
            _checkersSwitch, _lightSwitch;

        osg::ref_ptr<osg::Geode> _activeObject;
        osg::ref_ptr<osg::MatrixTransform> _geoRoot; // root of all non-GUI plugin geometry
        osg::ref_ptr<osg::PositionAttitudeTransform> _crosshairPat;
        osg::ref_ptr<osgText::Text> _scoreText; // GUI to display current score
        std::string _dataDir;
        bool _loaded; // whether the model has finished loading
        float _doorDist; // distance doors are currently translated
        int _activeDoor;
        int _lightColor, _mode;
        bool _doorInView;


        osg::ref_ptr<osg::Geometry> drawBox(osg::Vec3 center, float x, 
            float y, float z, osg::Vec4 color, float texScale = 1);

        osg::ref_ptr<osg::Geometry> makeQuad(float width, float height,
                osg::Vec4 color, osg::Vec3 pos);

};

};
#endif

