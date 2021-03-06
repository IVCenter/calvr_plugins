
#include <cvrKernel/SceneObject.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuRangeValue.h>
#include <cvrMenu/MenuCheckbox.h>

#include <osg/Uniform>
#include <cvrKernel/CVRPlugin.h>
#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/NodeMask.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrUtil/OsgMath.h>
#include <PluginMessageType.h>

#include <osg/Depth>
#include <osg/Material>
#include <osg/ShapeDrawable>

#include <iostream>
#include <cstdio>
#include <fstream>
#include <cmath>

#include <string>



class LightObject : public cvr::SceneObject
{
    public:
        LightObject(std::string name, std::string filename, osg::Quat pcRot, float pcScale, osg::Vec3 pcPos);
        virtual ~LightObject();

        void init(std::string name, std::string filename, osg::Quat pcRot, float pcScale, osg::Vec3 pcPos);


        void next();
        void previous();

        void setAlpha(float alpha);
        float getAlpha();

        void setRotate(float rotate);
        float getRotate();


        virtual void menuCallback(cvr::MenuItem * item);
        virtual void updateCallback(int handID, const osg::Matrix & mat);
        virtual bool eventCallback(cvr::InteractionEvent * ie);

        void preFrameUpdate();
        osg::ref_ptr<osg::LightSource>  _lightSource;

    protected:
        void updateZoom(osg::Matrix & mat);
        void startTransition();

        bool _printValues;
        bool _removeOnClick;
        bool _active;
        bool _loaded;
        bool _visible;
        

        cvr::MenuButton * loadMap;
        cvr::MenuButton * saveMap;
        cvr::MenuButton * saveNewMap;
        cvr::MenuButton * resetMap;
        cvr::MenuCheckbox * activeMap;
        cvr::MenuCheckbox * visibleMap;
        cvr::MenuCheckbox * pVisibleMap;
        cvr::MenuRangeValue * rxMap;
        cvr::MenuRangeValue * ryMap;
        cvr::MenuRangeValue * rzMap;

        int _zoomValuator;
        int _spinValuator;
        bool _sharedValuator;

};


