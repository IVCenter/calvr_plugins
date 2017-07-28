#include "NetworkWallSceneObject.h"

#include <cvrUtil/OsgMath.h>


NetworkWallSceneObject::NetworkWallSceneObject(std::string name, bool navigation,
        bool movable, bool clip, bool contextMenu, bool showBounds) :
        TiledWallSceneObject(name,false,movable,clip,contextMenu,showBounds)
{
    // set default scale
    _scaleIncrement = 0.05;
    _maxScale = 200.0f;
    _currentScale = 1.0;
}

NetworkWallSceneObject::~NetworkWallSceneObject()
{
}

bool NetworkWallSceneObject::processEvent(cvr::InteractionEvent * ie)
{
    if(!_tiledWallMovement)
    {
        return SceneObject::processEvent(ie);
    }

	// check for valuator
	cvr::ValuatorInteractionEvent * vie = ie->asValuatorEvent(); 

    if(vie)
    {
        osg::Matrix handMatrix = cvr::TrackingManager::instance()->getHandMat(vie->getHand());

        // cast ray to determine intersection point
        osg::Vec3 lineP1, lineP2(0,1000.0,0), planePoint,
                            planeNormal(0,-1,0), intersect;
        float w;

        // transform points in world space to scene object space
        lineP1 = lineP1 * handMatrix * getWorldToObjectMatrix();
        lineP2 = lineP2 * handMatrix * getWorldToObjectMatrix();

        if(cvr::linePlaneIntersectionRef(lineP1,lineP2,planePoint,
                planeNormal,intersect,w))
        {
            osg::Matrix intersectMatObj;
            intersectMatObj.setTrans(intersect);
            
            // get current scale
            float scale = getScale();

            // determine if scale up or down 
            if( vie->getValue() > 0 )
                _currentScale = 1.0 + _scaleIncrement;
            else if( vie->getValue() < 0 )
                _currentScale = 1.0 - _scaleIncrement;
                
            // make sure scale is not outside bound
            if( ( (_currentScale * scale) <= 1.0 ) || ( (_currentScale * scale) >= _maxScale ) )    
                _currentScale = 1.0;

            osg::Matrix scaleMat; 
            scaleMat.makeScale(_currentScale, _currentScale, _currentScale);

            // try directly setting the position  
            setTransform(osg::Matrix::inverse(intersectMatObj) * scaleMat * intersectMatObj * getTransform()); 
        }
	}

    return TiledWallSceneObject::processEvent(ie);
}
