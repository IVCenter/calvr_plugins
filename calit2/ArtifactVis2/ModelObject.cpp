#include "ModelObject.h"
//#include "PanoViewLOD.h"


//#define PRINT_TIMING

using namespace cvr;
using namespace std;
using namespace osg;

ModelObject::ModelObject(std::string name, std::string filename, osg::Quat pcRot, float pcScale, osg::Vec3 pcPos, std::map< std::string, osg::ref_ptr<osg::Node> > objectMap) : SceneObject(name,false,false,false,true,false)

{
    _active = false;
    _loaded = false;
    _visible = false;
    _shadow = true;
    _objectMap = objectMap;
    init(name,filename,pcRot,pcScale,pcPos);
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
	    setMovable(false);
	    addMoveMenuItem();
	    addNavigationMenuItem();
            float min = 0.0001;
            float max = 1;
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

	    activeMap = new MenuCheckbox("Active",true);
	    activeMap->setCallback(this);
	    addMenuItem(activeMap);
           // _pointClouds[i]->activeMap = mc;

            
	    visibleMap = new MenuCheckbox("Visible",true);
	    visibleMap->setCallback(this);
	    addMenuItem(visibleMap);

	    pVisibleMap = new MenuCheckbox("Panel Visible",true);
	    pVisibleMap->setCallback(this);
	   // addMenuItem(pVisibleMap);
 //           _query[q]->artifacts[inc]->model->pVisibleMap = mc;
           // _query[q]->artifacts[inc]->model->pVisible = true;

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
	        std::cerr << "Save." << std::endl;
                // saveModelConfig(_pointClouds[i], false);
	}
        else if (item == saveNewMap)
        {
	        std::cerr << "Save New." << std::endl;
                // saveModelConfig(_pointClouds[i], true);
	}
        else if (item == resetMap)
        {
	        std::cerr << "Reset." << std::endl;
               
	}
        else if (item == activeMap)
        {
            if (activeMap->getValue())
            {
	        std::cerr << "Active." << std::endl;
                 _active = true;
                 setMovable(true);
                 activeMap->setValue(true);
            }
            else
            {
                 _active = false;
                 setMovable(false);
                 activeMap->setValue(false);

	        std::cerr << "DeActive." << std::endl;
            }
	}
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
    if(_moving)
    {

      //osg::Vec3 orig = getPosition();
      //cerr << "So Pos: " << orig.x() << " " << orig.y() << " " << orig.z() << "\n";
      //printf("moving\n");
    }
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

void ModelObject::attachToScene(osgShadow::ShadowedScene* shadowRoot)
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
        if(shadowRoot != NULL)
        {
          //CVRPlugin * artifactVis2;
          //std::string plugin = "ArtifactVis2";
          //artifactVis2 = PluginManager::instance()->getPlugin(plugin);
        shadowRoot->addChild(_root);
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

void ModelObject::detachFromScene(osgShadow::ShadowedScene* shadowRoot)
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
        if(shadowRoot != NULL)
        {
        shadowRoot->removeChild(_root);
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
