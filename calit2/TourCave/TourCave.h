#ifndef TOUR_CAVE_PLUGIN_H
#define TOUR_CAVE_PLUGIN_H

#include <cvrKernel/CVRPlugin.h>

#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuButton.h>
#include <cvrUtil/MultiListenSocket.h>
#include <cvrUtil/CVRSocket.h>

#include <osg/Vec3>
#include <osg/Matrix>

#include <string>
#include <vector>

#define HAS_AUDIO 0

#ifdef HAS_AUDIO
#include "Audio/AssetManager.h"
#endif


using namespace cvr;

class TourCave : public CVRPlugin, public MenuCallback
{
    public:
        TourCave();
        ~TourCave();

        bool init();
        void preFrame();

        void menuCallback(MenuItem * item);

        void message(int type, char * & data, bool);

    protected:

        void checkSockets();
        bool processSocketInput(CVRSocket* socket);
        void processCommands(char *, int size);

        std::vector<char> _commandList;

	enum StopAction
        {
            STOP,
            PAUSE,
            NOTHING
        };

        enum StopLocation
        {
            PATH_END,
            NEXT_PATH
        };

        enum ModeStatus
        {
            STARTED,
            DONE
        };

        struct TourAudio
        {
            std::string name;
            double triggerTime;
            bool loop;
            bool started;
            osg::Vec3 pos;
            float distance;
            float volume;
            StopAction sa;
            StopLocation sl;
        };

        TourAudio * _backgroundAudio;

        std::vector<int> _modePathIDList;
        std::vector<float> _modePathSpeedList;

        std::vector<std::vector<TourAudio*> > _modeAudioList;

        int _currentMode;
        double _currentTime;

        SubMenu * _tourCaveMenu;
        std::vector<MenuButton *> _modeButtonList;

        osg::Matrix _startMat;
        float _startScale;

        MultiListenSocket * _mls;
        std::vector<CVRSocket *> _socketList;

        ModeStatus _mStatus;

#ifdef HAS_AUDIO
        AssetManager * _am;
#endif
};

#endif
