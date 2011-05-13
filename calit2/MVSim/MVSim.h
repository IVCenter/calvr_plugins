#ifndef PLUGIN_TEST
#define PLUGIN_TEST

#include <kernel/CVRPlugin.h>
#include <kernel/ScreenMVSimulator.h>
#include <menu/MenuButton.h>
#include <menu/SubMenu.h>
#include <menu/MenuCheckbox.h>
#include <menu/MenuRangeValue.h>
#include <menu/PopupMenu.h>

class MVSim : public cvr::CVRPlugin, public cvr::MenuCallback
{
    public:
        MVSim();
        ~MVSim();

        bool init();

        void menuCallback(cvr::MenuItem * item);

        void preFrame();

    protected:
        cvr::ScreenMVSimulator * _screenMVSim;
        osg::Matrix * head0;
        osg::Matrix * head1;

        osg::ref_ptr<osg::MatrixTransform> viewTransform0;
        osg::ref_ptr<osg::MatrixTransform> viewTransform1;

        void stepEvent();

        bool _run;
        float _delay;
        int _event;
        cvr::SubMenu * mvsMenu;
        cvr::MenuButton * startSim;
        cvr::MenuButton * stopSim;
        cvr::MenuButton * resetSim;
        cvr::MenuButton * stepSim;
        cvr::MenuButton * setHead1to0;
        cvr::MenuRangeValue * delaySim;
};

#endif
