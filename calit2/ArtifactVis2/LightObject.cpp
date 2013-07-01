#include "LightObject.h"
//#include "PanoViewLOD.h"


//#define PRINT_TIMING

using namespace cvr;
using namespace std;
using namespace osg;

LightObject::LightObject(std::string name, std::string filename, osg::Quat pcRot, float pcScale, osg::Vec3 pcPos) : SceneObject(name,false,false,false,true,false)
{
    _active = false;
    _loaded = false;
    _visible = false;
     osg::Vec4 lpos = Vec4(pcPos.x(),pcPos.y(),pcPos.z(),0);
    _lightSource = new osg::LightSource;
    _lightSource->getLight()->setPosition(lpos);
    _lightSource->getLight()->setAmbient(osg::Vec4(0.2, 0.2, 0.2, 1));
    _lightSource->getLight()->setDiffuse(osg::Vec4(0.8, 0.8, 0.8, 1));
    init(name,filename,pcRot,pcScale,pcPos);
}


LightObject::~LightObject()
{

}

void LightObject::init(std::string name, std::string filename, osg::Quat pcRot, float pcScale, osg::Vec3 pcPos)
{
    string type = filename;

    osg::Vec4f colorY = Vec4f(1, 1, 0, 1);
    Cone* cubeShape = new Cone(pcPos, 1.0, 2.0);
    float rotDegrees[3];
		rotDegrees[0] = DegreesToRadians(0.0);
		rotDegrees[1] = DegreesToRadians(0.0);
		rotDegrees[2] = DegreesToRadians(0.0);
		osg::Quat rot = osg::Quat(rotDegrees[0], osg::Vec3d(1,0,0),rotDegrees[1], osg::Vec3d(0,1,0),rotDegrees[2], osg::Vec3d(0,0,1)); 
    cubeShape->setRotation(rot);
    ShapeDrawable* shapeDrawable = new ShapeDrawable(cubeShape);
   // shapeDrawable->setTessellationHints(hints);
    shapeDrawable->setColor(colorY);
    osg::ref_ptr<osg::Geode> sphereGeode = new Geode();  
    sphereGeode->addDrawable(shapeDrawable);
        
            float currentScale = 1.0;
	    osg::Switch* switchNode = new osg::Switch();
	    addChild(switchNode);
	switchNode->addChild(sphereGeode.get());
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



osg::Quat currentRot = pcRot;
osg::Vec3 currentPos = pcPos;



osg::Vec3 orig = currentPos; 
cerr << "Light Pos: " << orig.x() << " " << orig.y() << " " << orig.z() << "\n";

 setPosition(currentPos);     
 setScale(currentScale);
 setRotation(currentRot);     

 _active = true;
 _loaded = true;
 _visible = true;


}



void LightObject::setRotate(float rotate)
{
}

float LightObject::getRotate()
{
    float angle;
    return angle;
}

void LightObject::menuCallback(cvr::MenuItem * item)
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
                 attachToScene();
		}
                 visibleMap->setValue(true);
            }
            else
            {
                 _active = false;
                 setMovable(false);
                 activeMap->setValue(false);
                 detachFromScene();
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

void LightObject::updateCallback(int handID, const osg::Matrix & mat)
{

    //std::cerr << "Update Callback." << std::endl;
    if(_moving)
    {

      //osg::Vec3 orig = getPosition();
      //cerr << "So Pos: " << orig.x() << " " << orig.y() << " " << orig.z() << "\n";
      //printf("moving\n");
  osg::Vec3 pos = getPosition();
  osg::Vec4 lpos = Vec4(pos.x(),pos.y(),pos.z(),0);
  _lightSource->getLight()->setPosition(lpos);
      //osg::Vec3 orig = getPosition();
//      cerr << "So Pos: " << pos.x() << " " << pos.y() << " " << pos.z() << "\n";
      //printf("moving\n");
    }
}

bool LightObject::eventCallback(cvr::InteractionEvent * ie)
{
    if(ie->asTrackedButtonEvent())
    {
	TrackedButtonInteractionEvent * tie = ie->asTrackedButtonEvent();

         if(_moving && (tie->getInteraction() == BUTTON_DRAG || tie->getInteraction() == BUTTON_UP))
{

}
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

void LightObject::preFrameUpdate()
{
}


