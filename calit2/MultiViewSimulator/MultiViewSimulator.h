#ifndef PLUGIN_TEST
#define PLUGIN_TEST

#include <kernel/CVRPlugin.h>
#include <menu/MenuButton.h>
#include <menu/SubMenu.h>
#include <menu/MenuCheckbox.h>
#include <menu/MenuRangeValue.h>
#include <menu/PopupMenu.h>

class MultiViewSimulator : public cvr::CVRPlugin, public cvr::MenuCallback
{
    public:
        MultiViewSimulator();
        ~MultiViewSimulator();

        bool init();

        void menuCallback(cvr::MenuItem * item);

        void preFrame();
        //void postFrame();
        //bool keyEvent(bool keyDown, int key, int mod);
        //bool buttonEvent(int type, int button, int hand, const osg::Matrix& mat);
        //bool mouseButtonEvent(int type, int button, int x, int y, const osg::Matrix& mat);

    protected:
        osg::Matrix * head0;
        osg::Matrix * head1;

        osg::ref_ptr<osg::MatrixTransform> viewTransform0;
        osg::ref_ptr<osg::MatrixTransform> viewTransform1;

        void stepEvent();

        bool _run;
        float _delay;
        int _event;
        cvr::SubMenu * mvsMenu;
        cvr::MenuCheckbox * multipleUsers;
        cvr::SubMenu * headMenu;
        cvr::MenuButton * startSim;
        cvr::MenuButton * stopSim;
        cvr::MenuButton * resetSim;
        cvr::MenuButton * stepSim;
        cvr::MenuButton * setHead1to0;
        cvr::MenuRangeValue * delaySim;
        cvr::SubMenu * contributionMenu;
        cvr::MenuCheckbox * linearFunc;
        cvr::MenuCheckbox * gaussianFunc;
        cvr::MenuCheckbox * orientation3d;
        cvr::SubMenu * zoneMenu;
        cvr::MenuCheckbox * autoAdjust;
        cvr::MenuRangeValue * zoneRowQuantity;
        cvr::MenuRangeValue * zoneColumnQuantity;
        cvr::MenuRangeValue * autoAdjustTarget;
        cvr::MenuRangeValue * autoAdjustOffset;
};

#endif
