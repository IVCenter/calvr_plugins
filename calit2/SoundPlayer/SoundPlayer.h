#ifndef _SOUNDPLAYER_
#define _SOUNDPLAYER_

#include <cvrKernel/CVRPlugin.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuRangeValue.h>
#include <cvrKernel/ComController.h>

#include <osg/MatrixTransform>

#include "SCServer.hpp"
#include "Sound.hpp"

#include <string>
#include <vector>
#include <map>

class SoundPlayer: public cvr::MenuCallback, public cvr::CVRPlugin
{
    public:        

        SoundPlayer();
        virtual ~SoundPlayer();

        bool init();
        void menuCallback(cvr::MenuItem * item);
        void preFrame();
        bool processEvent(cvr::InteractionEvent * event);

    protected:

        cvr::SubMenu * MLMenu, * loadMenu;
        cvr::MenuButton * removeButton;
        cvr::MenuRangeValue *_volumeSlider;

        // Sound
        std::vector<cvr::MenuButton*> _soundButtons;
        std::vector<sc::Sound *> _sounds;
        sc::SCServer * _AudioServer;

};

#endif
