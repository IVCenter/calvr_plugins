#ifndef OSG_PATH_RECORDER_H
#define OSG_PATH_RECORDER_H

#include <kernel/CVRPlugin.h>
#include <menu/SubMenu.h>
#include <menu/MenuButton.h>
#include <menu/MenuText.h>
#include <menu/MenuCheckbox.h>
#include <menu/MenuRangeValue.h>

#include <osg/AnimationPath>

#include <vector>
#include <string>

using namespace cvr;

class OsgPathRecorder : public CVRPlugin, public MenuCallback
{
    public:
        OsgPathRecorder();
        virtual ~OsgPathRecorder();

        bool init();

        void preFrame();
        void menuCallback(MenuItem * item);
        void message(int type, char * & data, bool);

    protected:

        void loadCurrentFile();
        void saveCurrentPath();

        enum OpMode
        {
            NONE = 0,
            RECORD,
            PLAYBACK
        };

        enum OpStatus
        {
            START = 0,
            PAUSE,
            STOP
        };

        SubMenu * _prMenu;
        SubMenu * _selectMenu;

        MenuRangeValue * _timeScaleRV;

        MenuText * _activeFile;
        MenuText * _timeText;
        MenuText * _pointsText;

        MenuCheckbox * _playbackCB;
        MenuCheckbox * _recordCB;
        MenuCheckbox * _realtimeCB;

        MenuButton * _startB;
        MenuButton * _pauseB;
        MenuButton * _stopB;
        MenuButton * _saveB;
        MenuButton * _emitB;
        MenuButton * _gotoFirstB;
        MenuButton * _gotoLastB;
        MenuButton * _removeLastB;

        std::vector<MenuButton *> _files;

        osg::AnimationPath * _path;

        double _time;
        double _emitTime;

        osg::Vec3d _lastPos;

	int _numPoints;
        double _timeScale;

        std::string _dataDir;
        std::string _currentFile;

        osg::Matrixd _lastTransform;
        double _lastScale;

        OpMode _mode;
        OpStatus _status;
};

#endif
