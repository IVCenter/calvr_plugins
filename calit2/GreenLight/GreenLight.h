#ifndef _GREEN_LIGHT_H_
#define _GREEN_LIGHT_H_

#include <list>
#include <set>
#include <vector>

#include <osg/AnimationPath>
#include <kernel/CVRPlugin.h>
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
                set<Node *> nodes; // node-sub-graph loaded in via osgDB readNodeFile
                AnimationStatus status; // status of animation
                double time; // time-point within animation path
                list<Entity *> group; // other entities that should animate together

                void handleAnimation();
                void beginAnimation();
                void addChild(Entity * child);

            protected:
                void createNodeSet(Node * node);
        };

        // Menu Items
        SubMenu * _glMenu;
        MenuCheckbox * _showBoxCheckbox;

        // Entities
        
        Entity * _box;          // box/frame
        vector<Entity *> _door; // doors
        Entity * _waterPipes;   // water pipes

        // Functions
        bool loadBox();
        bool handleIntersection(Node * iNode);
};

#endif
