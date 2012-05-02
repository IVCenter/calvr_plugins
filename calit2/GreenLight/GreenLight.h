#ifndef _GREEN_LIGHT_H_
#define _GREEN_LIGHT_H_

#include <list>
#include <set>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/SceneObject.h>
#include <cvrKernel/CVRPlugin.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuImage.h>
#include <cvrMenu/MenuList.h>
#include <cvrMenu/MenuText.h>
#include <cvrMenu/DialogPanel.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuSystem.h>
#include <cvrMenu/PopupMenu.h>

#include <osg/AnimationPath>
#include <osg/MatrixTransform>
#include <osg/ShapeDrawable>
#include <osg/Texture2D>
#include <osg/NodeVisitor>

#ifdef WITH_OSGEARTH
/*** OSG EARTH PLUGINS ***/

#include <osgEarth/Map>
#include <osgEarth/MapNode>
#include <osgEarth/FindNode>
#include <osgEarth/Utils>

#include <osgEarth/ElevationQuery>

#endif

#include <osgDB/FileUtils>
#include <osgDB/FileNameUtils>

/*************************/

/*** osgParticle things ***/
#include <osgParticle/ParticleSystem>
#include <osgParticle/Particle>
#include <osg/PositionAttitudeTransform>
/**************************/

/*** oasClientSound Things **********************/
#include "oasClient/OASSound.h"


/************************************************/

#include "Utility.h"

class GreenLight : public cvr::CVRPlugin, public cvr::MenuCallback
{
//  friend class SceneManager;


    public:
        GreenLight();
        ~GreenLight();

        bool init();

        void menuCallback(cvr::MenuItem * item);

        void preFrame();
        void postFrame();
        bool processEvent(cvr::InteractionEvent * event);
        bool buttonEvent(int type, int button, int hand, const osg::Matrix& mat);
        bool mouseButtonEvent(int type, int button, int x, int y, const osg::Matrix& mat);

        bool keyboardEvent(int key, int type);

        osg::MatrixTransform * InitSmoke();

        osg::ref_ptr<osg::MatrixTransform> OsgE_MT; // transform nodes of this entity
        osg::Matrixd output;

        osg::MatrixTransform * scaleMT;
        osg::MatrixTransform * pluginMT;
        osg::Matrixd*      scaleMatrix; 
        osg::Vec3d*        scaleVector; 

        osg::ref_ptr<osg::LOD> _glLOD;

#ifdef WITH_OSGEARTH
        osgEarth::Map * mapVariable;
#endif
    protected:

        class Component; // forward declaration

        class Entity
        {
            public:
                enum AnimationStatus { START, FORWARD, END, REVERSE };

                Entity(osg::Node * node, osg::Matrix mat = osg::Matrix::identity());

                osg::ref_ptr<osg::MatrixTransform> transform; // transform nodes of this entity
                osg::ref_ptr<osg::AnimationPath> path; // animation path (null if non-existent)
                osg::ref_ptr<osg::Node> mainNode;   // change mainNode to be the type of node with overriden accept.
             // TODO change nodes to type: set< ref_ptr< Node > >
                std::set<osg::Node *> nodes; // node-sub-graph loaded in via osgDB readNodeFile
                AnimationStatus status; // status of animation
                double time; // time-point within animation path
                std::list<Entity *> group; // other entities that should animate together

                void handleAnimation();
                void beginAnimation();
                void addChild(Entity * child);
                void showVisual(bool show);
                virtual void setTransparency(bool transparent);
                virtual void setColor(const osg::Vec3 color);
                virtual void defaultColor();

                virtual Component * asComponent() {return NULL;}

            protected:
                void createNodeSet(osg::Node * node);
        };

        class Component : public Entity
        {
            public:
                Component(osg::Geode * geode, std::string componentName, osg::Matrix mat = osg::Matrix::identity());

                std::string name;
                bool selected;
                int minWattage; // used for coloring range
                int maxWattage; // used for coloring range
                std::string cluster;

                void setDefaultMaterial();
                void setTransparency(bool transparent);
                void setColor(const osg::Vec3 color);
                void setColor(const std::list<osg::Vec3> colors);
                void defaultColor();
                bool select(bool select);

                Component * asComponent() {return this;}

                static osg::ref_ptr<osg::Uniform> _displayTexturesUni;
                static osg::ref_ptr<osg::Uniform> _neverTextureUni;

                // Variables for animation.
                int animationPosition;
                bool animating;
                osg::Geode * animationMarker;

                void playSound();
                oasclient::OASSound * soundComponent;

            protected:
                osg::ref_ptr<osg::Texture2D> _colors;
                osg::ref_ptr<osg::Image> _data;
                osg::ref_ptr<osg::Uniform> _alphaUni;
                osg::ref_ptr<osg::Uniform> _colorsUni;
        };

        typedef struct {
            std::string name;
            int rack;
            int slot;
            int height;
         } Hardware;

/***************** LOD SWITCHING MECHANISM **********/
        static int lodLevel;
        osg::MatrixTransform * secondDegreeMT; // used as the second LOD?

        // used in LoadEntities?
        class NodeA : public osg::Node
        {
            public:
                virtual void accept(osg::NodeVisitor&);
        };

        class MTA : public osg::MatrixTransform
        {
            public:
                virtual void accept(osg::NodeVisitor&);
                int LLOD;
        };
/*********************************************************/

/****** PARTICLE SYSTEM VARIABLES ***************************/
        osgParticle::ParticleSystem * _osgParticleSystem;
        osgParticle::Particle _pTemplate;

/****** MISCELLANEOUS VARIABLES *****************************/
        // animation function for power comsumption.
        void animatePower();
        bool osgEarthInit;
/****** END: MISCELLANEOUS VARIABLES ************************/

        cvr::SubMenu * _glMenu;
        cvr::MenuCheckbox * _showSceneCheckbox;

        cvr::SubMenu * _hardwareSelectionMenu;
        cvr::MenuCheckbox * _selectionModeCheckbox;
        cvr::SubMenu * _selectClusterMenu;
        std::set< cvr::MenuCheckbox * > _clusterCheckbox;
        cvr::MenuButton * _selectAllButton;
        cvr::MenuButton * _deselectAllButton;

        cvr::SubMenu * _displayComponentsMenu;
        cvr::MenuCheckbox * _xrayViewCheckbox;
        cvr::MenuCheckbox * _displayFrameCheckbox;
        cvr::MenuCheckbox * _displayDoorsCheckbox;
        cvr::MenuCheckbox * _displayWaterPipesCheckbox;
        cvr::MenuCheckbox * _displayElectricalCheckbox;
        cvr::MenuCheckbox * _displayFansCheckbox;
        cvr::MenuCheckbox * _displayRacksCheckbox;
        cvr::MenuCheckbox * _displayComponentTexturesCheckbox;

        cvr::SubMenu * _powerMenu;
        cvr::MenuButton * _loadPowerButton;
        cvr::MenuCheckbox * _pollHistoricalDataCheckbox;
        cvr::MenuCheckbox * _displayPowerCheckbox;
        cvr::MenuCheckbox * _magnifyRangeCheckbox;
        cvr::MenuText * _legendText;
        cvr::MenuImage * _legendGradient;
        cvr::MenuText * _legendTextOutOfRange;
        cvr::MenuImage * _legendGradientOutOfRange;

        cvr::DialogPanel * _hoverDialog;

        // Timestamps
        cvr::SubMenu * _timeFrom;
        cvr::SubMenu * _timeTo;

        cvr::MenuText * _yearText;
        cvr::MenuText * _monthText;
        cvr::MenuText * _dayText;
        cvr::MenuText * _hourText;
        cvr::MenuText * _minuteText;

        cvr::MenuList * _yearFrom;
        cvr::MenuList * _monthFrom;
        cvr::MenuList * _dayFrom;
        cvr::MenuList * _hourFrom;
        cvr::MenuList * _minuteFrom;

        cvr::MenuList * _yearTo;
        cvr::MenuList * _monthTo;
        cvr::MenuList * _dayTo;
        cvr::MenuList * _hourTo;
        cvr::MenuList * _minuteTo;

        cvr::MenuButton * _navigateToPluginButton;
        cvr::MenuButton * _restorePreviousViewButton;

        // Entities
        Entity * _box;                     // box/frame
        std::vector<Entity *> _door;       // doors
        Entity * _waterPipes;              // water pipes
        Entity * _electrical;              // electrical
        Entity * _fans;                    // fans
        std::vector<Entity *> _rack;       // racks
        std::set<Component *> _components; // components in the racks

        // Additional Entity Info
        Entity * _mouseOver;
        Entity * _wandOver;
        std::map< std::string, std::set< Component * > * > _cluster;

        // File contents -- read/write via master node, copy to slave nodes via messages
        std::string _hardwareContents;
        std::string _powerContents;

        // Shaders
        osg::ref_ptr<osg::Program> _shaderProgram;

        // Functions
        bool loadScene();
        bool handleIntersection(osg::Node * iNode);
        void parseHardwareFile();
        void setPowerColors(bool displayPower);
        void selectComponent(Component * comp, bool select);
        void selectCluster(std::set< Component * > * cluster, bool select);
        void handleHoverOver(osg::Matrix pointerMat, Entity *& hovered, bool showHover);
        void doHoverOver(Entity *& last, Entity * current, bool showHover);
        osg::ref_ptr<osg::Geode> makeComponentGeode(float height, std::string textureFile = "");
        osg::Vec3 wattColor(float watt, int minWatt, int maxWatt);
        void createTimestampMenus();

        void InitializeOASClient();
    
};

#endif
