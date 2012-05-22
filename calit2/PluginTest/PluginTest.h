#ifndef PLUGIN_TEST
#define PLUGIN_TEST

#include <cvrKernel/CVRPlugin.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuRangeValue.h>
#include <cvrMenu/PopupMenu.h>
#include <cvrMenu/MenuTextButtonSet.h>
#include <cvrMenu/TabbedDialogPanel.h>
#include <cvrMenu/MenuScrollText.h>
#include <cvrKernel/SceneObject.h>
#include <cvrUtil/PointsNode.h>

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
        void createPointsNode();
        void testMulticast();

        osg::ref_ptr<osg::MatrixTransform> _pointsMT;
        osg::ref_ptr<cvr::PointsNode> _pointsNode;

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

        cvr::MenuScrollText * _mst;

        bool _loading;
        int _job;
};

#endif
