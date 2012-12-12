#ifndef BINAURAL_REALITY_PLUGIN_H
#define BINAURAL_REALITY_PLUGIN_H

#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/FileHandler.h>

#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuTextButtonSet.h>
#include <cvrMenu/MenuRangeValue.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/ScrollingDialogPanel.h>
#include <cvrMenu/MenuText.h>


class BinauralReality: public cvr::CVRPlugin, public cvr::MenuCallback
{
    public:
        BinauralReality();
        virtual ~BinauralReality();
        static BinauralReality * instance();
        bool init();
        void menuCallback(cvr::MenuItem * item);
        void preFrame();
        bool processEvent(cvr::InteractionEvent * event);

    protected:
        static BinauralReality * _myPtr;
        cvr::SubMenu *_BinauralRealityMenu;

};


#endif
