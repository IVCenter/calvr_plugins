#ifndef _PANOVIEW360_PLUGIN_H
#define _PANOVIEW360_PLUGIN_H

#include <osg/Geometry>
#include <osg/Texture2D>
#include <osg/Geometry>
#include <osg/Vec3>
#include <osg/Vec2>
#include <osg/Vec4>
#include <osg/StateSet>
#include <osg/StateAttribute>
#include <osg/PrimitiveSet> 
#include <osg/Matrix>
#include <osg/Switch>
#include <osg/MatrixTransform>
#include <string.h>
#include <osgUtil/IntersectVisitor>
#include <iostream>
#include "CylinderDrawable.h"
#include "SphereDrawable.h"
#include <menu/SubMenu.h>
#include <menu/MenuButton.h>
#include <menu/MenuRangeValue.h>
#include <kernel/CVRPlugin.h>

using namespace cvr;

class PanoView360 : public CVRPlugin, public MenuCallback
{
    public:
        PanoView360();
        virtual ~PanoView360();

        void menuCallback(MenuItem* item);

        bool processEvent(InteractionEvent * event);

        bool init();
        void preFrame();
        void parseConfig(std::string file);

    protected:
        void createLoadMenu(std::string tagBase, std::string tag, SubMenu * menu);

        SubMenu* _panoViewMenu;
        SubMenu* _loadMenu;
        MenuButton* _remove;
        MenuRangeValue* _tilesp;
        MenuRangeValue* _radiusp;
        MenuRangeValue* _viewanglep;
        MenuRangeValue* _camHeightp;
        MenuRangeValue* _viewanglepb;

        std::vector<MenuButton*> _menufilelist;

        osg::MatrixTransform * _root;
        PanoDrawable * _cdLeft;
        PanoDrawable * _cdRight;

        MenuItem * _nextLoad;
        bool _deleteWait;

        enum drawableShape
        {
            CYLINDER,
            SPHERE
        };

        struct loadinfo
        {
            std::string name;
            std::string right_eye_file;
            std::string left_eye_file;
            float radius;
            float viewangleh;
            float viewanglev;
            float camHeight;
            int segments;
            int texture_size;
            int flip;
            drawableShape shape;
            
        };

        std::vector<struct loadinfo *> _pictures;
        int _wasinit;

        std::map<std::string, std::map<int, std::vector<std::pair<std::pair<int, int>, int> > > > _eyeMap;

        std::string _configFile;

        bool _joystickSpin;
        float _spinScale;
};

#endif
