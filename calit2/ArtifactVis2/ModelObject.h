

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
#include <cvrKernel/PluginManager.h>
#include <cvrUtil/OsgMath.h>
#include <cvrUtil/TextureVisitors.h>
#include <PluginMessageType.h>

#include <osg/Depth>
#include <osgDB/ReadFile>
#include <osgDB/FileUtils>
#include <osg/CullFace>
#include <osg/TexEnv>
#include <osg/GLExtensions>
#include <osg/Material>
#include <osg/TextureCubeMap>
#include <osg/Texture2D>
#include <osgShadow/ShadowedScene>
#include <osgShadow/SoftShadowMap>
#include <osgShadow/ShadowMap>
/*
//osgBullet
#include <osgbDynamics/RigidBody.h>
#include <osgbDynamics/MotionState.h>
#include <osgbDynamics/GroundPlane.h>
#include <osgbCollision/CollisionShapes.h>
#include <osgbCollision/RefBulletObject.h>
#include <osgbCollision/Utils.h>
#include <osgbDynamics/TripleBuffer.h>
#include <osgbDynamics/PhysicsThread.h>
#include <osgbInteraction/DragHandler.h>
#include <osgbInteraction/LaunchHandler.h>
#include <osgbInteraction/SaveRestoreHandler.h>

#include <osgwTools/Shapes.h>

#include <btBulletDynamicsCommon.h>
//..............................................
*/
#include <iostream>
#include <cstdio>
#include <fstream>
#include <cmath>

#include <string>


class ModelObject : public cvr::SceneObject
{
    public:
        ModelObject(std::string name, std::string fullpath, std::string filename, std::string path, std::string filetype, std::string type, std::string group, osg::Quat pcRot, float pcScale, osg::Vec3 pcPos, std::map< std::string, osg::ref_ptr<osg::Node> > objectMap,osgShadow::ShadowedScene* shadowRoot);
        virtual ~ModelObject();

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
        virtual void attachToScene();
        virtual void detachFromScene();

        void preFrameUpdate();
        std::map< std::string, osg::ref_ptr<osg::Node> > _objectMap;

            bool _picked;
            bool firstPick;
            osg::Matrix lastHandMat;
            osg::Matrix lastHandInv;
            osg::Matrix lastobj2world;
/*
//            osg::MatrixTransform* rootPhysics;
	    btDynamicsWorld* dw;
	    btRigidBody* body;
            btPoint2PointConstraint* _constraint;
            osgbDynamics::MotionState* _constrainedMotionState;
*/
         void processMove(osg::Matrix & mat);
         void updateDragging();
         osgShadow::ShadowedScene* _shadowRoot;

	std::string _name;
	std::string _path;
	std::string _filename;
	std::string _q_filetype;
	std::string _q_type;
	std::string _q_group;
	osg::Vec3 _pos;
	osg::Quat _rot;
	float _scaleFloat;
	osg::Vec3 _posOrig;
	osg::Quat _rotOrig;
	float _scaleFloatOrig;

    protected:
        void updateZoom(osg::Matrix & mat);
        void startTransition();

        bool _printValues;
        bool _removeOnClick;
        bool _active;
        bool _loaded;
        bool _visible;
        bool _shadow;

        cvr::MenuButton * loadMap;
        cvr::MenuButton * saveMap;
        cvr::MenuButton * saveNewMap;
        cvr::MenuButton * resetMap;
        cvr::MenuCheckbox * shadowMap;
        cvr::MenuCheckbox * bbMap;
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


