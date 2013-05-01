#ifndef ELEVATORROOM_PLUGIN_H
#define ELEVATORROOM_PLUGIN_H


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

#include "AudioHandler.h"
#include "ModelHandler.h"
//#include "OAS/OASClient.h"

namespace ElevatorRoom
{

#define NUM_DOORS 8
#define FLASH_SPEED 4
#define NUM_ALLY_FLASH 3
#define NUM_ALIEN_FLASH 8
#define LIGHT_PAUSE_LENGTH 5

class ElevatorRoom: public cvr::CVRPlugin, public cvr::MenuCallback
{
    public:
        ElevatorRoom();
        virtual ~ElevatorRoom();
        static ElevatorRoom * instance();
        bool init();
        void menuCallback(cvr::MenuItem * item);
        void preFrame();
        bool processEvent(cvr::InteractionEvent * event);

    protected:
        enum GameMode
        {
            ONE,
            TWO,
            THREE,
            FOUR,
            FIVE
        };

        enum Phase
        {
            PAUSE,
            FLASHNEUTRAL,
            DOORCOLOR,
            OPENINGDOOR,
            DOOROPEN,
            CLOSINGDOOR
        };
 
        void loadModels();
        void clear();
        void chooseGameParameters(int &door, Mode &mode, bool &switched);
        void sendChar(unsigned char c);
        void dingTest();
        void turnLeft();
        void turnRight();
        void shoot();
        void flashAvatars();

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

        static ElevatorRoom * _myPtr;
        AudioHandler * _audioHandler; 
        ModelHandler * _modelHandler;

        cvr::SubMenu * _elevatorMenu, *_optionsMenu;
        cvr::MenuButton * _loadButton, * _clearButton;
        cvr::MenuRangeValue *_timeScaleRV;
        cvr::MenuText * _chancesText;
        cvr::MenuCheckbox *_dingCheckbox, *_pauseCB;

        std::map<std::string, cvr::MenuCheckbox*> _levelMap;

        osg::ref_ptr<osg::MatrixTransform> _geoRoot; // root of all non-GUI plugin geometry
        
        // Timing 
        float _startTime, _pauseTime, _flashStartTime, _timeScale;
        float _avatarFlashPerSec, _lightFlashPerSec, _checkSpeed, _doorFlashSpeed;
        float _pauseMin, _pauseMax, _flashNeutralMin, _flashNeutralMax,
        _solidColorMin, _solidColorMax, _doorOpenMin, _doorOpenMax;
        float _valEventTime, _valEventCutoff;

        float _modelScale; // scale of entire scene
        int _flashCount; // number of times active avatar has flashed
        int _activeDoor; // which door is currently opening/closing
        int _score; // current score (should be > 0)
        int _sockfd; // for EOG syncer communication
        int _alienChance, _allyChance, _checkerChance, _errorChance;

        bool _loaded, _paused, _switched; // whether the model has finished loading
        bool _hit, _noResponse; // whether a hit has been made on the active avatar
        bool _debug; // turns on debug messages to command line
        bool _connected; // for EOG syncer communication
        bool _soundEnabled;
        bool _valEvent;

        Mode _mode; // which kind of avatar is currently active
        Phase _phase;
        
        int _trialPhase, _trialCount;
        float _trialPauseTime;
        bool _trialPause;
        std::vector<int> _trialCounts;
        std::vector<float> _trialPauseLengths;
        std::vector<std::string> _trialThemes;

        // Config options
        bool _staticMode, _staticDoor, _doorMovement, _rotateOnly;

        osg::Quat _eventRot;
        osg::Vec3 _eventPos;
        osg::PositionAttitudeTransform *_headsoundPAT, *_handsoundPAT;

        float _transMult, _rotMult;
        float _transcale, _rotscale;
        
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

