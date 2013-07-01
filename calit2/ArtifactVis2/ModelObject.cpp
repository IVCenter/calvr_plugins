#include "ModelObject.h"

#include <ConvertTools.h>
using namespace cvr;
using namespace std;
using namespace osg;

ModelObject::ModelObject(std::string name, std::string fullpath,std::string filename, std::string path, std::string filetype, std::string type, std::string group, osg::Quat pcRot, float pcScale, osg::Vec3 pcPos, std::map< std::string, osg::ref_ptr<osg::Node> > objectMap,osgShadow::ShadowedScene* shadowRoot) : SceneObject(name,false,false,false,true,false)

{
    _active = false;
    _loaded = false;
    _visible = false;
    _shadow = false;
   _shadowRoot = shadowRoot;
    _objectMap = objectMap;

//For Saving
_name = name;
_path = path;
_filename = filename;
_q_filetype = filetype;
_q_type = type;
_q_group = group;
_pos = pcPos;
_rot = pcRot;
_scaleFloat = pcScale;

//For Reset
_posOrig = pcPos;
_rotOrig = pcRot;
_scaleFloatOrig = pcScale;

//For Bullet Physics
//_constraint = NULL;
// _constrainedMotionState = NULL;
// dw = bw;
_picked = false;
firstPick = true;

    init(name,fullpath,pcRot,pcScale,pcPos);
}


ModelObject::~ModelObject()
{

}

void ModelObject::init(std::string name, std::string filename, osg::Quat pcRot, float pcScale, osg::Vec3 pcPos)
{
 string currentModelPath = filename;

         Vec3 currentPos = pcPos;
        Quat  currentRot = pcRot;
  //Check if ModelPath has been loaded
  Node* modelNode;
  
            if (_objectMap.count(currentModelPath) == 0)
	    {
		 modelNode = osgDB::readNodeFile(currentModelPath);
	    }
            else
            {
            modelNode = _objectMap[currentModelPath];
            }
  
//Add Lighting and Culling

		if(false)
		{
		    osg::StateSet* stateset = modelNode->getOrCreateStateSet();
		    //stateset->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
		    stateset->setMode(GL_LIGHTING, osg::StateAttribute::ON);
		}
		if(true)
		{
		    osg::StateSet * stateset = modelNode->getOrCreateStateSet();
		    osg::CullFace * cf=new osg::CullFace();
		    //cf->setMode(osg::CullFace::BACK);
		    cf->setMode(osg::CullFace::FRONT_AND_BACK);
		    stateset->setAttributeAndModes( cf, osg::StateAttribute::OFF | osg::StateAttribute::OVERRIDE);
		}
                if(false)
		{
		TextureResizeNonPowerOfTwoHintVisitor tr2v(false);
		modelNode->accept(tr2v);
                }
                if(true)
                {
		TextureResizeNonPowerOfTwoHintVisitor tr2v(false);
		modelNode->accept(tr2v);
                    StateSet* ss = modelNode->getOrCreateStateSet();
                    ss->setMode(GL_LIGHTING, StateAttribute::ON | osg::StateAttribute::OVERRIDE);
                    Material* mat = new Material();
                    mat->setColorMode(Material::AMBIENT_AND_DIFFUSE);
		    bool rgb_config = false;
		    float r,g,b,a;
                    r = g = b = a = 1;
		    if(rgb_config)
                    {
			r = 51.0/255.0;
			g = 25.0/255.0;
			b = 0/255.0;
			a = 255.0/255.0;

                    }
                    Vec4 color_dif(r, g, b, a);
                    mat->setDiffuse(Material::FRONT_AND_BACK, color_dif);
                    ss->setAttribute(mat);
                    ss->setAttributeAndModes(mat, osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE);
                }
        
            float currentScale = pcScale;
	    osg::Switch* switchNode = new osg::Switch();
	    addChild(switchNode);
//Add currentNode to switchNode
	switchNode->addChild(modelNode);
//Add menu system
	    setNavigationOn(true);
	    setMovable(true);
	    addMoveMenuItem();
	    addNavigationMenuItem();
            float min = 0.0001;
            float max = 10;
            addScaleMenuItem("Scale",min,max,currentScale);
	    SubMenu * sm = new SubMenu("Position");
	    addMenuItem(sm);

	    loadMap = new MenuButton("Load");
	    loadMap->setCallback(this);
	    sm->addItem(loadMap);

	    SubMenu * savemenu = new SubMenu("Save");
	    sm->addItem(savemenu);

	    saveMap = new MenuButton("Save");
	    saveMap->setCallback(this);
	    savemenu->addItem(saveMap);

	    saveNewMap = new MenuButton("Save New Kml");
	    saveNewMap->setCallback(this);
	    savemenu->addItem(saveNewMap);

	    resetMap = new MenuButton("Reset to Origin");
	    resetMap->setCallback(this);
	    addMenuItem(resetMap);

	    shadowMap = new MenuCheckbox("Shadowing",false);
	    shadowMap->setCallback(this);
	    addMenuItem(shadowMap);

	    bbMap = new MenuCheckbox("Bounding Box",false);
	    bbMap->setCallback(this);
	    addMenuItem(bbMap);

	    activeMap = new MenuCheckbox("Active",true);
	    activeMap->setCallback(this);
	    addMenuItem(activeMap);

	    visibleMap = new MenuCheckbox("Visible",true);
	    visibleMap->setCallback(this);
	    addMenuItem(visibleMap);

	    pVisibleMap = new MenuCheckbox("Panel Visible",true);
	    pVisibleMap->setCallback(this);

            float rValue = 0;
            min = -1;
            max = 1;
            rxMap = new MenuRangeValue("rx",min,max,rValue);
            rxMap->setCallback(this);
	    addMenuItem(rxMap);

            ryMap = new MenuRangeValue("ry",min,max,rValue);
            ryMap->setCallback(this);
	    addMenuItem(ryMap);

            rzMap = new MenuRangeValue("rz",min,max,rValue);
            rzMap->setCallback(this);
	    addMenuItem(rzMap);



osg::Vec3 orig = currentPos; 
cerr << "Pos: " << orig.x() << " " << orig.y() << " " << orig.z() << "\n";

 setPosition(currentPos);     
 setScale(currentScale);
 setRotation(currentRot);     

 _active = true;
 _loaded = true;
 _visible = true;



/*
            m = getTransform();
	    btCollisionShape* cs = osgbCollision::btConvexTriMeshCollisionShapeFromOSG(_root);
	    osgbDynamics::MotionState* motion = new osgbDynamics::MotionState();
	    motion->setTransform( rootPhysics );
	   // motion->setTransform( _root );
	    motion->setParentTransform( m );
	    btScalar mass( 1. );
	    btVector3 inertia( 0, 0, 0 );
	    cs->calculateLocalInertia( mass, inertia );
	    btRigidBody::btRigidBodyConstructionInfo rb( mass, motion, cs, inertia );


	    body = new btRigidBody( rb );
	    body->setActivationState( DISABLE_DEACTIVATION );
	    bw->addRigidBody( body );

            srh->add(_name, body );
            //bulletRoot->addChild(rootPhysics);
           // bulletRoot->addChild(_root);
            shadowRoot->addChild(rootPhysics);
    updateMatrices();

    _attached = true;
*/
}
void ModelObject::setRotate(float rotate)
{
}

float ModelObject::getRotate()
{
    float angle;
    return angle;
}

void ModelObject::menuCallback(cvr::MenuItem * item)
{
        if (item == saveMap)
        {
	        //std::cerr << "Save." << std::endl;
                _pos = getPosition();
                _rot = getRotation();
                _scaleFloat = getScale();
                ConvertTools* convertTools = new ConvertTools("test");
                convertTools->saveModelConfig(_name,_path,_filename,_q_filetype,_q_type,_q_group,_pos,_rot,_scaleFloat, false);
	}
        else if (item == saveNewMap)
        {
	        //std::cerr << "Save New." << std::endl;
                _pos = getPosition();
                _rot = getRotation();
                _scaleFloat = getScale();
                ConvertTools* convertTools = new ConvertTools("test");
                convertTools->saveModelConfig(_name,_path,_filename,_q_filetype,_q_type,_q_group,_pos,_rot,_scaleFloat, true);
	}
        else if (item == resetMap)
        {
	        std::cerr << "Reset." << std::endl;
                setPosition(_posOrig);
                setRotation(_rotOrig);
                setScale(_scaleFloatOrig);
               
	}
        else if (item == shadowMap)
        {
	    //std::cerr << "Shadow." << std::endl;
            if (shadowMap->getValue())
            {
              detachFromScene();
              _shadow = true;
              attachToScene();
	    }
	    else
	    {

              detachFromScene();
              _shadow = false;
              attachToScene();

	    }
	}
        else if (item == bbMap)
        {
	    //std::cerr << "Bounding Box." << std::endl;
            if (bbMap->getValue())
            {
             _root->addChild(_boundsTransform);
	    }
	    else
	    {
             _root->removeChild(_boundsTransform);
	    }

	}
        else if (item == activeMap)
        {
            if (activeMap->getValue())
            {
	        //std::cerr << "Active." << std::endl;
                 _active = true;
                 setMovable(true);
                 activeMap->setValue(true);
            }
            else
            {
                 _active = false;
                 setMovable(false);
                 activeMap->setValue(false);

	        //std::cerr << "DeActive." << std::endl;
            }
	}
        /*
        else if (item == visibleMap)
        {
            if (visibleMap->getValue())
            {
	        std::cerr << "Visible." << std::endl;
                 _active = true;
                if(!_visible)
                {
                 //attachToScene();
		}
                 visibleMap->setValue(true);
            }
            else
            {
                 _active = false;
                 setMovable(false);
                 activeMap->setValue(false);
                 //detachFromScene();
                 _visible = false;
	        std::cerr << "NotVisible." << std::endl;
            }
	}
        */
        else if (item == rxMap)
        {
	        //std::cerr << "Rotate." << std::endl;
                osg::Quat mSo = getRotation();
                osg::Quat mRot;
                float deg = rxMap->getValue();
                if(rxMap->getValue() > 0)
                {
		  mRot = osg::Quat(0.05, osg::Vec3d(1,0,0),0, osg::Vec3d(0,1,0),0, osg::Vec3d(0,0,1)); 
                }
                else
                {
		  mRot = osg::Quat(-0.05, osg::Vec3d(1,0,0),0, osg::Vec3d(0,1,0),0, osg::Vec3d(0,0,1)); 
                }
                mSo *= mRot;
                setRotation(mSo);
                rxMap->setValue(0);
	}
        else if (item == ryMap)
        {
	        //std::cerr << "Rotate." << std::endl;
                osg::Quat mSo = getRotation();
                osg::Quat mRot;
                float deg = ryMap->getValue();
                if(ryMap->getValue() > 0)
                {
		  mRot = osg::Quat(0, osg::Vec3d(1,0,0),0.05, osg::Vec3d(0,1,0),0, osg::Vec3d(0,0,1)); 
                }
                else
                {
		  mRot = osg::Quat(0, osg::Vec3d(1,0,0),-0.05, osg::Vec3d(0,1,0),0, osg::Vec3d(0,0,1)); 
                }
                mSo *= mRot;
                 setRotation(mSo);
                ryMap->setValue(0);
	}
        else if (item == rzMap)
        {
	       // std::cerr << "Rotate." << std::endl;
                osg::Quat mSo = getRotation();
                osg::Quat mRot;
                float deg = rzMap->getValue();
                if(rzMap->getValue() > 0)
                {
		  mRot = osg::Quat(0, osg::Vec3d(1,0,0),0, osg::Vec3d(0,1,0),0.05, osg::Vec3d(0,0,1)); 
                }
                else
                {
		  mRot = osg::Quat(0, osg::Vec3d(1,0,0),0, osg::Vec3d(0,1,0),-0.05, osg::Vec3d(0,0,1)); 
                }
                mSo *= mRot;
                 setRotation(mSo);
                rzMap->setValue(0);
	}

    SceneObject::menuCallback(item);
}

void ModelObject::updateCallback(int handID, const osg::Matrix & mat)
{

    //std::cerr << "Update Callback." << std::endl;
    /*
    if(_moving)
    {

      //osg::Vec3 orig = getPosition();
      //cerr << "So Pos: " << orig.x() << " " << orig.y() << " " << orig.z() << "\n";
     // printf("moving\n");
      updateDragging();
    }
    else
    {
      if(_picked)
      {
        _picked = false;
        dw->removeConstraint( _constraint );
        delete _constraint;
        _constraint = NULL;
        _constrainedMotionState = NULL;
        firstPick = true;
       
      }
    }
   */
}

bool ModelObject::eventCallback(cvr::InteractionEvent * ie)
{
    if(ie->asTrackedButtonEvent())
    {
	TrackedButtonInteractionEvent * tie = ie->asTrackedButtonEvent();


	if(tie->getButton() == 0 && tie->getInteraction() == BUTTON_DOWN)
	{
	   // printf("button 0\n");
	    //return true;
	}
	if(tie->getButton() == 1 && tie->getInteraction() == BUTTON_DOWN)
	{
	   // printf("button 1\n");
	   // return true;
	}
	/*if(tie->getButton() == 0 && tie->getInteraction() == BUTTON_DOWN)
	{
	    updateZoom(tie->getTransform());

	    return true;
	}
	if(tie->getButton() == 0 && (tie->getInteraction() == BUTTON_DRAG || tie->getInteraction() == BUTTON_UP))
	{
	    float val = -PluginHelper::getValuator(0,1);
	    if(fabs(val) > 0.25)
	    {
		_currentZoom += val * _zoomScale * PluginHelper::getLastFrameDuration() * 0.25;
		if(_currentZoom < -2.0) _currentZoom = -2.0;
		if(_currentZoom > 0.5) _currentZoom = 0.5;
	    }

	    updateZoom(tie->getTransform());

	    return true;
	}*/
	if(tie->getButton() == 4 && tie->getInteraction() == BUTTON_DOWN)
	{
	    return true;
	}
    }
    else if(ie->asKeyboardEvent())
    {
    }
    else if(ie->asValuatorEvent())
    {
	//std::cerr << "Valuator id: " << ie->asValuatorEvent()->getValuator() << " value: " << ie->asValuatorEvent()->getValue() << std::endl;

	ValuatorInteractionEvent * vie = ie->asValuatorEvent();
	if(vie->getValuator() == _spinValuator)
	{
	    if(true)
	    {
		float val = vie->getValue();
		if(fabs(val) < 0.15)
		{
		    return true;
		}

		if(val > 1.0)
		{
		    val = 1.0;
		}
		else if(val < -1.0)
		{
		    val = -1.0;
		}

		if(val < 0)
		{
		    val = -(val * val);
		}
		else
		{
		    val *= val;
		}

		return true;
	    }
	}
	if(vie->getValuator() == _zoomValuator)
	{
		return true;
	}
    }
    return false;
}

void ModelObject::preFrameUpdate()
{
}

void ModelObject::attachToScene()
{
    if(_attached)
    {
        return;
    }

    if(!_registered)
    {
        std::cerr << "Scene Object: " << _name
                << " must be registered before it is attached." << std::endl;
        return;
    }

    if(_parent)
    {
        std::cerr << "Scene Object: attachToScene: error, " << _name
                << " is a child object." << std::endl;
        return;
    }

    if(_navigation)
    {
        if(_shadow)
        {
/*
            osg::Matrix m(getTransform());
            rootPhysics = new osg::MatrixTransform( m );
            rootPhysics->addChild(_root);
	    btCollisionShape* cs = osgbCollision::btConvexTriMeshCollisionShapeFromOSG(_root);
	    osgbDynamics::MotionState* motion = new osgbDynamics::MotionState();
	   // motion->setTransform( rootPhysics );
	    motion->setTransform( _root );
	   // motion->setParentTransform( m );
	    btScalar mass( 2. );
	    btVector3 inertia( 0, 0, 0 );
	    cs->calculateLocalInertia( mass, inertia );
	    btRigidBody::btRigidBodyConstructionInfo rb( mass, motion, cs, inertia );


	    body = new btRigidBody( rb );
	    body->setActivationState( DISABLE_DEACTIVATION );
	    bw->addRigidBody( body );

            srh->add(_name, body );
            //bulletRoot->addChild(rootPhysics);
           // bulletRoot->addChild(_root);
*/
            _shadowRoot->addChild(_root);
        }
        else
        {
        SceneManager::instance()->getObjectsRoot()->addChild(_root);
        }
    }
    else
    {
        SceneManager::instance()->getScene()->addChild(_root);
    }

    updateMatrices();

    _attached = true;
}

void ModelObject::detachFromScene()
{
    if(!_attached)
    {
        return;
    }

    if(SceneManager::instance()->getMenuOpenObject() == this)
    {
        SceneManager::instance()->closeOpenObjectMenu();
    }

    if(_navigation)
    {
        if(_shadow == true)
        {
        _shadowRoot->removeChild(_root);
        }
        else
        {
        SceneManager::instance()->getObjectsRoot()->removeChild(_root);
	}
    }
    else
    {
        SceneManager::instance()->getScene()->removeChild(_root);
    }

    _attached = false;
}
void ModelObject::processMove(osg::Matrix & mat)
{
   // std::cerr << "Process move." << std::endl;
   /*
    osg::Matrix m;
    if(getNavigationOn())
    {
        m = PluginHelper::getWorldToObjectTransform();
    }
    rootPhysics->setMatrix(_lastobj2world * _lastHandInv * mat * m * _root2obj);

    splitMatrix();
*/
  //  _lastHandMat = mat;
  //  _lastHandInv = osg::Matrix::inverse(mat);
 //   _lastobj2world = getObjectToWorldMatrix();
   // updateDragging();
}
void ModelObject::updateDragging()
{
/*
 if(firstPick)
 {

    osg::Matrix mat = currentHand;
    lastHandMat = mat;
    lastHandInv = osg::Matrix::inverse(mat);
    lastobj2world = getObjectToWorldMatrix();
    firstPick=false;

 }
 else
 {
    osg::Matrix mat = currentHand;
    osg::MatrixTransform* pointM = new osg::MatrixTransform();
    osg::Matrix wo = PluginHelper::getWorldToObjectTransform();
    osg::Matrix handInv = osg::Matrix::inverse(currentHand);
    pointM->setMatrix(lastobj2world * lastHandInv * mat * wo * _root2obj);
    osg::Vec3 pointOnPlane = pointM->getMatrix().getTrans();
   // std::cerr << pointOnPlane.x() << " " << pointOnPlane.y() << " " << pointOnPlane.z() << "\n";
    lastHandMat = mat;
    lastHandInv = osg::Matrix::inverse(mat);
    lastobj2world = getObjectToWorldMatrix();
   // pointOnPlane = Vec3(-pointOnPlane.x(),pointOnPlane.z(),-pointOnPlane.y());
   // std::cerr << pointOnPlane.x() << " " << pointOnPlane.y() << " " << pointOnPlane.z() << "\n";
if(_constrainedMotionState == NULL)
{
    _constrainedMotionState = dynamic_cast< osgbDynamics::MotionState* >( body->getMotionState() );
    osg::Matrix ow2col;
        ow2col = _constrainedMotionState->computeOsgWorldToCOLocal();
    osg::Vec3d pickPointBulletOCLocal = pointOnPlane * ow2col;
    
    _constraint = new btPoint2PointConstraint( *body,
        osgbCollision::asBtVector3( pickPointBulletOCLocal ) );
    dw->addConstraint( _constraint );
    _picked = true;
   // std::cerr << "Drag move." << std::endl;
}
else
{
        osg::Matrix ow2bw;
        if( _constrainedMotionState != NULL )
            ow2bw = _constrainedMotionState->computeOsgWorldToBulletWorld();
        osg::Vec3d bulletPoint = pointOnPlane * ow2bw;

        _constraint->setPivotB( osgbCollision::asBtVector3( bulletPoint ) );
   // std::cerr << "Drag moving." << std::endl;
}
}
*/
}
