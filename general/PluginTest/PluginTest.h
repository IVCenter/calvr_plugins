#ifndef PLUGIN_TEST
#define PLUGIN_TEST

#include <kernel/CVRPlugin.h>
#include <menu/MenuButton.h>
#include <menu/SubMenu.h>
#include <menu/MenuCheckbox.h>
#include <menu/MenuRangeValue.h>
#include <menu/PopupMenu.h>
#include <menu/MenuTextButtonSet.h>
#include <menu/TabbedDialogPanel.h>
#include <kernel/SceneObject.h>

class PluginTest : public cvr::CVRPlugin, public cvr::MenuCallback
{
    public:
        PluginTest();
        ~PluginTest();

        bool init();

        void menuCallback(cvr::MenuItem * item);

        void preFrame();
        void postFrame();
    protected:
        void createSphereTexture();
        void testMulticast();

        cvr::MenuButton * testButton1;
        cvr::MenuButton * testButton2;
        cvr::MenuButton * testButton3;
        cvr::MenuButton * testButton4;
        cvr::MenuButton * testButton5;

        cvr::MenuTextButtonSet * textButtonSet1;

        cvr::MenuCheckbox * checkbox1;

        cvr::MenuRangeValue * rangeValue;

        cvr::SubMenu * menu1;
        cvr::SubMenu * menu2;
        cvr::SubMenu * menu3;

        cvr::PopupMenu * popup1;
        cvr::SubMenu * pmenu1;
        cvr::MenuCheckbox * pcheckbox1;
        cvr::MenuButton * pbutton1;

        cvr::TabbedDialogPanel * tdp1;

        cvr::SceneObject * _testobj;
        cvr::SceneObject * _testobj2;

        bool _loading;
        int _job;
};

#endif
