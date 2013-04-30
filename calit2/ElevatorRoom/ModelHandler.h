#ifndef ELEVATOR_MODELHANDLER_H
#define ELEVATOR_MODELHANDLER_H 

#include <cvrKernel/PluginHelper.h>
#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/ComController.h>
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
#include "AudioHandler.h"

#define NUM_DOORS 8
#define DOOR_SPEED 0.040
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

enum Colors
{
    WHITE,
    RED,
    BLUE,
    ORANGE,
    YELLOW,
    GREEN,
    BROWN,
    GREY
};

class ModelHandler
{ public:
        ModelHandler();
        ~ModelHandler();
        
        void update();
        void loadModels(osg::MatrixTransform* root);
        void openDoor();
        void closeDoor();
        void setAudioHandler(AudioHandler * handler);

        void turnLeft();
        void turnRight();

        void flashActiveLight();
        void flashCheckers();
        void flashAlien();
        void flashAlly();

        void setMode(Mode mode);
        void setActiveDoor(int doorNum);
        void setSwitched(bool switched);

        void setScore(int score);
        void setAlien(bool val);
        void setAlly(bool val);
        void setLight(bool val);

        void setLevel(std::string level);
        void clear();

        osg::ref_ptr<osg::Geode> getActiveObject();
        float getDoorDistance();
        bool doorInView();

    protected:
        std::vector<osg::ref_ptr<osg::PositionAttitudeTransform> > _leftdoorPat, _rightdoorPat;
        std::vector<osg::ref_ptr<osg::ShapeDrawable> > _lights;
        
        std::vector<osg::ref_ptr<osg::Geode> > _walls, _elevators, _floors, 
            _doors, _ceilings;

        // child 1 non-flashing, child 2 flashing
        std::vector<osg::ref_ptr<osg::Switch> > _aliensSwitch, _alliesSwitch, 
            _checkersSwitch, _lightSwitch, _leftdoorSwitch;

        std::string _wallTex, _floorTex, _ceilingTex, _doorTex,
            _alienTex, _allyTex, _checkTex1, _checkTex2, _elevTex;

        osg::ref_ptr<osg::Geode> _activeObject;
        osg::ref_ptr<osg::MatrixTransform> _geoRoot, _root; // root of all non-GUI plugin geometry
        osg::ref_ptr<osg::PositionAttitudeTransform> _crosshairPat;
        osg::ref_ptr<osgText::Text> _scoreText; // GUI to display current score
        std::string _dataDir;
        bool _loaded; // whether the model has finished loading
        float _doorDist; // distance doors are currently translated
        int _activeDoor, _viewedDoor;
        float _totalAngle;
        int _lightColor;
        bool _doorInView, _switched, _turningLeft, _turningRight;
        Mode _mode;

        AudioHandler * _audioHandler;

        std::vector<osg::Vec4> _colors;

        osg::ref_ptr<osg::Geometry> drawBox(osg::Vec3 center, float x, 
            float y, float z, osg::Vec4 color, float texScale = 1);

        osg::ref_ptr<osg::Geometry> makeQuad(float width, float height,
                osg::Vec4 color, osg::Vec3 pos);

};

};
#endif

