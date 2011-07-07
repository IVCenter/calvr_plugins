#ifndef _GREEN_LIGHT_H_
#define _GREEN_LIGHT_H_

#include <list>
#include <set>
#include <vector>

#include <osg/AnimationPath>
#include <config/ConfigManager.h>
#include <kernel/CVRPlugin.h>
#include <menu/MenuButton.h>
#include <menu/MenuCheckbox.h>
#include <menu/SubMenu.h>

#include <osg/MatrixTransform>

using namespace cvr;
using namespace std;
using namespace osg;

class GreenLight : public CVRPlugin, public MenuCallback
{
    public:
        GreenLight();
        ~GreenLight();

        bool init();

        void menuCallback(MenuItem * item);

        void preFrame();
        void postFrame();
        bool keyEvent(bool keyDown, int key, int mod);
        bool buttonEvent(int type, int button, int hand, const osg::Matrix& mat);
        bool mouseButtonEvent(int type, int button, int x, int y, const osg::Matrix& mat);

    protected:
 
        class Entity
        {
            public:
                enum AnimationStatus { START, FORWARD, END, REVERSE };

                Entity(Node * node, Matrix mat = Matrix::identity());

                ref_ptr<MatrixTransform> transform; // transform nodes of this entity
                ref_ptr<AnimationPath> path; // animation path (null if non-existent)
                ref_ptr<Node> mainNode;
// TODO change nodes to type: set< ref_ptr< Node > >
                set<Node *> nodes; // node-sub-graph loaded in via osgDB readNodeFile
                AnimationStatus status; // status of animation
                double time; // time-point within animation path
                list<Entity *> group; // other entities that should animate together

                void handleAnimation();
                void beginAnimation();
                void addChild(Entity * child);
                void showVisual(bool show);
                void setColor(const Vec3 color);
                void setTransparency(bool transparent);
                void setDefaultMaterial();

            protected:
                void createNodeSet(Node * node);
        };

        typedef struct {
            string name;
            int rack;
            int slot;
            int height;
         } Hardware;

        // Menu Items
        SubMenu * _glMenu;
        MenuCheckbox * _showSceneCheckbox;

        SubMenu * _displayComponentsMenu;
        MenuCheckbox * _componentsViewCheckbox;
        MenuCheckbox * _displayFrameCheckbox;
        MenuCheckbox * _displayDoorsCheckbox;
        MenuCheckbox * _displayWaterPipesCheckbox;
        MenuCheckbox * _displayElectricalCheckbox;
        MenuCheckbox * _displayFansCheckbox;
        MenuCheckbox * _displayRacksCheckbox;

        SubMenu * _powerMenu;
        MenuCheckbox * _displayPowerCheckbox;
        MenuButton * _loadPowerButton;

        // Entities
        Entity * _box;          // box/frame
        vector<Entity *> _door; // doors
        Entity * _waterPipes;   // water pipes
        Entity * _electrical;   // electrical
        Entity * _fans;         // fans
        vector<Entity *> _rack; // racks
        map<string,Entity *> _components; // mapping of component names to component entities

        // File contents -- read/write via master node, copy to slave nodes via messages
        string _hardwareContents;
        string _powerContents;

        // Functions
        bool loadScene();
        bool handleIntersection(Node * iNode);
        void parseHardwareFile();
        void downloadFile(string downloadUrl, string fileName, string &content);
        void setPowerColors(bool displayPower);
};

#endif
