#include "MultiHandInteraction.h"

#include <kernel/InteractionManager.h>
#include <kernel/SceneManager.h>
#include <input/TrackingManager.h>
#include <kernel/PluginHelper.h>

#include <cmath>

using namespace cvr;
using namespace std;

CVRPLUGIN(MultiHandInteraction)

MultiHandInteraction::MultiHandInteraction()
{
    _interactionStarted = false;

    _refUpdated = false;
    _activeUpdated = false;
    _setLastRefHand  = false;
}

MultiHandInteraction::~MultiHandInteraction()
{
}

bool MultiHandInteraction::init()
{
    if(TrackingManager::instance()->getNumHands() < 2)
    {
	std::cerr << "MulitHandInteraction init failed: At least two hands are required." << std::endl;
	return false;
    }
    return true;
}

void MultiHandInteraction::preFrame()
{
    if(_interactionStarted && _setLastRefHand)
    {
	_lastRefHandMat = TrackingManager::instance()->getHandMat(_refHand);
	_currentRefHandMat = _lastRefHandMat;
	//_refHandMat = TrackingManager::instance()->getHandMat(_refHand);
    }
}

bool MultiHandInteraction::processEvent(InteractionEvent * event)
{
    TrackedButtonInteractionEvent * tie = event->asTrackedButtonEvent();
    //std::cerr << "Button event hand: " << hand << " button: " << button << std::endl;
    if(!tie || tie->getButton())
    {
	return false;
    }

    /*if(type == BUTTON_DOWN)
    {
	std::cerr << "Button: " << button << " hand: " << hand << std::endl;
    }*/

    /*if(!_interactionStarted && type != BUTTON_DOWN)
    {
	return true;
    }*/

    if(!_interactionStarted && tie->getInteraction() == BUTTON_DOWN)
    {
	_activeHand = tie->getHand();
	if(tie->getHand())
	{
	    _refHand = 0;
	}
	else
	{
	    _refHand = 1;
	}

	//_refHandMat = TrackingManager::instance()->getHandMat(_refHand);
	//_activeHandMat = mat;
	_lastRefHandMat = TrackingManager::instance()->getHandMat(_refHand);
	_currentRefHandMat = _lastRefHandMat;
	_lastActiveHandMat = tie->getTransform();
	_currentActiveHandMat = _lastActiveHandMat;
	_navMode = Navigation::instance()->getPrimaryButtonMode();

	_refUpdated = false;
	_activeUpdated = false;
	_setLastRefHand = true;

	_interactionStarted = true;

	_startXForm = SceneManager::instance()->getObjectTransform()->getMatrix();
	/*
	switch(_navMode)
	{
	    case WALK:
	    case DRIVE:
		_startPoint = _activeHandMat.getTrans();
		_startRot = _refHandMat.getRotate();
		break;
	    case FLY:
		_startPoint = _activeHandMat.getTrans();
		_startRot = _refHandMat.getRotate();
		break;
	    case MOVE_WORLD:
		break;
	    case SCALE:
		_startScale = SceneManager::instance()->getObjectScale();
		_scalePoint = _refHandMat.getTrans() + ((_activeHandMat.getTrans() - _refHandMat.getTrans()) / 2.0);
		_scalePoint = (_scalePoint * osg::Matrix::inverse(_startXForm)) / _startScale;
		_startScaleLength = (_activeHandMat.getTrans() - _refHandMat.getTrans()).length();
                //std::cerr << "ScalePoint x: " << _scalePoint.x() << " y: " << _scalePoint.y() << " z: " << _scalePoint.z() << std::endl;
                //std::cerr << "Scale length: " << _startScaleLength << std::endl;
		break;
	    case NONE:
	    default:
		return true;
	}*/

	return true;
    }
    /*else if(hand != _activeHand)
    {
	return true;
    }*/

    /*if(type == BUTTON_UP)
    {
        std::cerr << "Button up event." << std::endl;
    }
    else if(type == BUTTON_DRAG)
    {
        std::cerr << "Button drag event." << std::endl;
    }*/

    if(tie->getInteraction() == BUTTON_UP || tie->getInteraction() == BUTTON_DRAG)
    {
	if(tie->getHand() == _activeHand)
	{
	    //std::cerr << "Primary hand updated." << std::endl;
	    _lastActiveHandMat = _currentActiveHandMat;
	    _currentActiveHandMat = tie->getTransform();
	    _activeUpdated = true;
	}
	else if(tie->getHand() == _refHand)
	{
	    //std::cerr << "Ref hand updated." << std::endl;
	    _lastRefHandMat = _currentRefHandMat;
	    _currentRefHandMat = tie->getTransform();
	    if(_setLastRefHand)
	    {
		switch(_navMode)
		{
		    case WALK:
		    case DRIVE:
			_startPoint = _currentRefHandMat.getTrans() + ((_currentActiveHandMat.getTrans() - _currentRefHandMat.getTrans()) / 2.0);
			break;
		    case FLY:
			_startPoint = _currentRefHandMat.getTrans() + ((_currentActiveHandMat.getTrans() - _currentRefHandMat.getTrans()) / 2.0);;
			break;
		    case MOVE_WORLD:
			break;
		    case SCALE:
			_startScale = SceneManager::instance()->getObjectScale();
			_scalePoint = _currentRefHandMat.getTrans() + ((_currentActiveHandMat.getTrans() - _currentRefHandMat.getTrans()) / 2.0);
			_scalePoint = (_scalePoint * osg::Matrix::inverse(_startXForm)) / _startScale;
			_startScaleLength = (_currentActiveHandMat.getTrans() - _currentRefHandMat.getTrans()).length();
			break;
		    case NONE:
		    default:
			return true;
		}
		_setLastRefHand = false;
	    }
	    _refUpdated = true;
	}

	if(_activeUpdated && _refUpdated)
	{
	    newProcessNav();
	    //_lastRefHandMat = _currentRefHandMat;
	    //_lastActiveHandMat = _currentActiveHandMat;
	    _refUpdated = _activeUpdated = false;
	}
    }

    if(tie->getHand() == _activeHand && tie->getInteraction() == BUTTON_UP)
    {
	_interactionStarted = false;
    }
    else if(tie->getHand() == _refHand && tie->getInteraction() == BUTTON_UP)
    {
	_setLastRefHand = true;
    }

    return true;
}

void MultiHandInteraction::processNav()
{
    switch(_navMode)
    {
	case WALK:
	case DRIVE:
	{
	    osg::Vec3 offset = -(_activeHandMat.getTrans() - _startPoint) / 10.0;
	    osg::Matrix m;
            
            osg::Matrix r;
            r.makeRotate(_startRot);
            osg::Vec3 pointInit = osg::Vec3(0,1,0);
            pointInit = pointInit * r;
            pointInit.z() = 0.0;

            r.makeRotate(_refHandMat.getRotate());
            osg::Vec3 pointFinal = osg::Vec3(0,1,0);
            pointFinal = pointFinal * r;
            pointFinal.z() = 0.0;

            osg::Matrix turn;
            if(pointInit.length2() > 0 && pointFinal.length2() > 0)
            {
                pointInit.normalize();
                pointFinal.normalize();
                float dot = pointInit * pointFinal;
                float angle = acos(dot) / 15.0;
                if(dot > 1.0 || dot < -1.0)
                {
                    angle = 0.0;
                }
                else if( (pointInit ^ pointFinal).z() < 0)
                {
                    angle = -angle;
                }
                turn.makeRotate(-angle,osg::Vec3(0,0,1));
            }

	    osg::Matrix objmat = PluginHelper::getObjectTransform()->getMatrix();
            osg::Vec3 origin = _refHandMat.getTrans();
            //origin = mat.getTrans() - origin;
            m.makeTranslate(offset + origin);
	    m = objmat * osg::Matrix::translate(-origin) * turn * m;
	    PluginHelper::setObjectMatrix(m);
	    break;
	}
	case FLY:
	{
	    osg::Matrix rotOffset = osg::Matrix::rotate(_startRot.inverse()) * osg::Matrix::rotate(_refHandMat.getRotate());
            osg::Quat rot = rotOffset.getRotate();
            rot = rot.inverse();
            double angle;
            osg::Vec3 vec;
            rot.getRotate(angle,vec);
            rot.makeRotate(angle / 20.0,vec);
            rotOffset.makeRotate(rot);
	    osg::Vec3 posOffset = (_activeHandMat.getTrans() - _startPoint) / 20.0;
	    osg::Matrix objmat = PluginHelper::getObjectTransform()->getMatrix();
	    osg::Vec3 origin = _refHandMat.getTrans();
	    objmat = (objmat * osg::Matrix::translate(-origin) * rotOffset * osg::Matrix::translate(origin - posOffset));
	    PluginHelper::setObjectMatrix(objmat);
	    break;
	}
	case MOVE_WORLD:
	{
	    //osg::MatrixTransform * mt = SceneManager::instance()->getObjectTransform();
	    //mt->setMatrix(_startXForm * osg::Matrix::translate(-_eventPos) * osg::Matrix::rotate(_eventRot.inverse()) * mat);
	    break;
	}
	case SCALE:
	{
	    osg::Vec3 pos1 = _activeHandMat.getTrans();
	    osg::Vec3 pos2 = _refHandMat.getTrans();
	    float diff = (pos1 - pos2).length();
	    float newScale = (diff / _startScaleLength) * _startScale;
	    SceneManager::instance()->setObjectScale(newScale);
            //std::cerr << "New Scale is: " << newScale << std::endl;
            osg::Matrix objmat = PluginHelper::getObjectTransform()->getMatrix();
	    osg::Vec3 offset = -((_scalePoint * newScale * objmat) - (_scalePoint * _startScale * objmat));
	    osg::Matrix m;
	    m.makeTranslate(offset);
	    m = _startXForm * m;
	    PluginHelper::setObjectMatrix(m);
	    break;
	}
	case NONE:
	default:
	    break;
    }
}

void MultiHandInteraction::newProcessNav()
{
    switch(_navMode)
    {
	case WALK:
	case DRIVE:
	{
	    osg::Vec3 lastAPoint = _lastActiveHandMat.getTrans();
	    osg::Vec3 lastRPoint = _lastRefHandMat.getTrans();
	    osg::Vec3 currentAPoint = _currentActiveHandMat.getTrans();
	    osg::Vec3 currentRPoint = _currentRefHandMat.getTrans();

	    osg::Vec3 midpoint = ((lastAPoint - lastRPoint) / 2.0) + lastRPoint;
	    osg::Vec3 diffvec = _startPoint - midpoint;
	    diffvec = diffvec / 10.0;

	    osg::Vec3 rotAxis = (currentAPoint - lastAPoint) ^ (currentRPoint - lastRPoint);
	    rotAxis.normalize();
	    //std::cerr << "Axis of Rot x: " << rotAxis.x() << " y: " << rotAxis.y() << " z: " << rotAxis.z() << std::endl;
	    if(rotAxis.isNaN() || !rotAxis.valid() )
	    {
		return;
	    }

	    osg::Vec3 mp2lastA = lastAPoint - midpoint;
	    mp2lastA.z() = 0;
	    mp2lastA.normalize();
	    osg::Vec3 mp2currentA = currentAPoint - midpoint;
	    mp2currentA.z() = 0;
	    mp2currentA.normalize();
	    osg::Vec3 mp2lastR = lastRPoint - midpoint;
	    mp2lastR.z() = 0;
	    mp2lastR.normalize();
	    osg::Vec3 mp2currentR = currentRPoint - midpoint;
	    mp2currentR.z() = 0;
	    mp2currentR.normalize();
	    osg::Matrix ma1,ma2;
	    ma1.makeRotate(mp2currentA, mp2lastA);
	    ma2.makeRotate(mp2currentR, mp2lastR);
	    /*double a1 = mp2lastA * mp2currentA;
	    double a2 = mp2lastR * mp2currentR;
	    if(a1 > 1.0 || a1 < -1.0)
	    {
		a1 = 1.0;
	    }
	    if(a2 > 1.0 || a2 < -1.0)
	    {
		a1 = 1.0;
	    }
	    double angle = acos(a1) +
	                   acos(a2);

	    angle /= 10.0;
	    if(std::isnan(angle))
	    {
		return;
	    }
	    std::cerr << "Angle: " << angle << std::endl;*/
	    osg::Matrix mat = PluginHelper::getObjectMatrix();
	    //mat = mat * osg::Matrix::translate(-midpoint) * osg::Matrix::rotate(angle, rotAxis) * osg::Matrix::translate(midpoint);
	    mat = mat * osg::Matrix::translate(-midpoint) * ma1 * ma2 * osg::Matrix::translate(midpoint + diffvec);
	    PluginHelper::setObjectMatrix(mat);

	    break;
	}
	case FLY:
	{
	    osg::Vec3 lastAPoint = _lastActiveHandMat.getTrans();
	    osg::Vec3 lastRPoint = _lastRefHandMat.getTrans();
	    osg::Vec3 currentAPoint = _currentActiveHandMat.getTrans();
	    osg::Vec3 currentRPoint = _currentRefHandMat.getTrans();

	    osg::Vec3 midpoint = ((lastAPoint - lastRPoint) / 2.0) + lastRPoint;
	    osg::Vec3 diffvec = _startPoint - midpoint;
	    diffvec = diffvec / 10.0;

	    osg::Vec3 rotAxis = (currentAPoint - lastAPoint) ^ (currentRPoint - lastRPoint);
	    rotAxis.normalize();
	    //std::cerr << "Axis of Rot x: " << rotAxis.x() << " y: " << rotAxis.y() << " z: " << rotAxis.z() << std::endl;
	    if(rotAxis.isNaN() || !rotAxis.valid() )
	    {
		return;
	    }

	    osg::Vec3 mp2lastA = lastAPoint - midpoint;
	    mp2lastA.normalize();
	    osg::Vec3 mp2currentA = currentAPoint - midpoint;
	    mp2currentA.normalize();
	    osg::Vec3 mp2lastR = lastRPoint - midpoint;
	    mp2lastR.normalize();
	    osg::Vec3 mp2currentR = currentRPoint - midpoint;
	    mp2currentR.normalize();
	    osg::Matrix ma1,ma2;
	    ma1.makeRotate(mp2currentA, mp2lastA);
	    ma2.makeRotate(mp2currentR, mp2lastR);

	    osg::Vec3 currentADir = osg::Vec3(0,1.0,0);
	    osg::Vec3 currentRDir = osg::Vec3(0,1.0,0);

	    currentADir = (currentADir * _currentActiveHandMat) - _currentActiveHandMat.getTrans();
	    currentRDir = (currentRDir * _currentRefHandMat) - _currentRefHandMat.getTrans();


	    //get current midpoint
	    osg::Vec3 currentMidpoint = ((_currentActiveHandMat.getTrans() - _currentRefHandMat.getTrans()) * 0.5) + _currentRefHandMat.getTrans();

	    //get plane normal
	    osg::Vec3 currentNormal = _currentActiveHandMat.getTrans() - currentMidpoint;
	    currentNormal.normalize();
	   
	    //std::cerr << "Normal x: " << currentNormal.x() << " y: " << currentNormal.y() << " z: " << currentNormal.z() << std::endl;
	    
	    osg::Vec3 activeProj = currentNormal ^ currentADir ^ currentNormal;
	    osg::Vec3 refProj = currentNormal ^ currentRDir ^ currentNormal;
	    activeProj.normalize();
	    refProj.normalize();

	    //check if valid
	    if(activeProj.length() < 0.9 || refProj.length() < 0.9)
	    {
		std::cerr << "Invalid proj" << std::endl;
		return;
	    }

	    osg::Quat projQuat;
	    projQuat.makeRotate(activeProj, refProj);

	    osg::Vec3 projAxis;
	    double projAngle;
	    projQuat.getRotate(projAngle, projAxis);

	    //std::cerr << "Angle: " << projAngle << std::endl;
	    //std::cerr << "ProjAxis x: " << projAxis.x() << " y: " << projAxis.y() << " z: " << projAxis.z() << std::endl;

	    if(projAngle < 0.0)
	    {
		// should never get here
		std::cerr << "Proj Angle < 0." << std::endl;
	    }

	    if(projAngle < 0.3)
	    {
		projAngle = 0;
	    }
	    else
	    {
		projAngle -= 0.3;
		projAngle *= 0.15;
		projAngle *= projAngle;
	    }

	    osg::Matrix projRot;
	    projRot.makeRotate(projAngle, projAxis);


	    /*double a1 = mp2lastA * mp2currentA;
	    double a2 = mp2lastR * mp2currentR;
	    if(a1 > 1.0 || a1 < -1.0)
	    {
		a1 = 1.0;
	    }
	    if(a2 > 1.0 || a2 < -1.0)
	    {
		a1 = 1.0;
	    }
	    double angle = acos(a1) +
	                   acos(a2);

	    angle /= 10.0;
	    if(std::isnan(angle))
	    {
		return;
	    }
	    std::cerr << "Angle: " << angle << std::endl;*/
	    osg::Matrix mat = PluginHelper::getObjectMatrix();
	    //mat = mat * osg::Matrix::translate(-midpoint) * osg::Matrix::rotate(angle, rotAxis) * osg::Matrix::translate(midpoint);
	    mat = mat * osg::Matrix::translate(-midpoint) * ma1 * ma2 * projRot * osg::Matrix::translate(midpoint + diffvec);
	    PluginHelper::setObjectMatrix(mat);

	    break;
	}
	case MOVE_WORLD:
	{
	    osg::Vec3 lastAPoint = _lastActiveHandMat.getTrans();
	    osg::Vec3 lastRPoint = _lastRefHandMat.getTrans();
	    osg::Vec3 currentAPoint = _currentActiveHandMat.getTrans();
	    osg::Vec3 currentRPoint = _currentRefHandMat.getTrans();

	    osg::Vec3 midpoint = ((lastAPoint - lastRPoint) / 2.0) + lastRPoint;

	    osg::Vec3 rotAxis = (currentAPoint - lastAPoint) ^ (currentRPoint - lastRPoint);
	    rotAxis.normalize();
	    //std::cerr << "Axis of Rot x: " << rotAxis.x() << " y: " << rotAxis.y() << " z: " << rotAxis.z() << std::endl;
	    if(rotAxis.isNaN() || !rotAxis.valid() )
	    {
		return;
	    }

	    osg::Vec3 mp2lastA = lastAPoint - midpoint;
	    mp2lastA.normalize();
	    osg::Vec3 mp2currentA = currentAPoint - midpoint;
	    mp2currentA.normalize();
	    osg::Vec3 mp2lastR = lastRPoint - midpoint;
	    mp2lastR.normalize();
	    osg::Vec3 mp2currentR = currentRPoint - midpoint;
	    mp2currentR.normalize();
	    osg::Matrix ma1,ma2;
	    ma1.makeRotate(mp2currentA, mp2lastA);
	    ma2.makeRotate(mp2currentR, mp2lastR);
	    

	    osg::Vec3 currentADir = osg::Vec3(0,1.0,0);
	    osg::Vec3 currentRDir = osg::Vec3(0,1.0,0);

	    currentADir = (currentADir * _currentActiveHandMat) - _currentActiveHandMat.getTrans();
	    currentRDir = (currentRDir * _currentRefHandMat) - _currentRefHandMat.getTrans();


	    //get current midpoint
	    osg::Vec3 currentMidpoint = ((_currentActiveHandMat.getTrans() - _currentRefHandMat.getTrans()) * 0.5) + _currentRefHandMat.getTrans();

	    //get plane normal
	    osg::Vec3 currentNormal = _currentActiveHandMat.getTrans() - currentMidpoint;
	    currentNormal.normalize();
	   
	    //std::cerr << "Normal x: " << currentNormal.x() << " y: " << currentNormal.y() << " z: " << currentNormal.z() << std::endl;
	    
	    osg::Vec3 activeProj = currentNormal ^ currentADir ^ currentNormal;
	    osg::Vec3 refProj = currentNormal ^ currentRDir ^ currentNormal;
	    activeProj.normalize();
	    refProj.normalize();

	    //check if valid
	    if(activeProj.length() < 0.9 || refProj.length() < 0.9)
	    {
		std::cerr << "Invalid proj" << std::endl;
		return;
	    }

	    osg::Quat projQuat;
	    projQuat.makeRotate(activeProj, refProj);

	    osg::Vec3 projAxis;
	    double projAngle;
	    projQuat.getRotate(projAngle, projAxis);

	    //std::cerr << "Angle: " << projAngle << std::endl;
	    //std::cerr << "ProjAxis x: " << projAxis.x() << " y: " << projAxis.y() << " z: " << projAxis.z() << std::endl;

	    projAngle *= 0.01;

	    osg::Matrix projRot;
	    projRot.makeRotate(projAngle, projAxis);

	    /*float angle = refProj * activeProj;
	    if(angle < -1.0 || angle > 1.0)
	    {
		std::cerr << "Invalid angle" << std::endl;
		return;
	    }
	    angle = acos(angle);*/
	    //std::cerr << "Angle: " << angle << std::endl;
/*
	    // find projected last active hand direction vector
	    osg::Vec3 lastdir = osg::Vec3(0,1.0,0);
	    lastdir = (lastdir * _lastActiveHandMat) - _lastActiveHandMat.getTrans();
	    lastdir.normalize();
	    osg::Vec3 normal = mp2lastA;
	    osg::Vec3 lastProj = normal ^ lastdir ^ normal;
	    lastProj.normalize();

	    // find projected current active hand direction vector
	    osg::Vec3 currentdir = osg::Vec3(0,1.0,0);
	    currentdir = (currentdir * _currentActiveHandMat) - _currentActiveHandMat.getTrans();
	    currentdir.normalize();
	    normal = mp2currentA;
	    osg::Vec3 currentProj = normal ^ lastdir ^ normal;
	    currentProj.normalize();

	    // check if vectors are valid
	    if(lastProj.length() < 0.9 || currentProj.length() < 0.9)
	    {
		return;
	    }

	    // more currect direction vector to same plane as last vector
	    currentProj = currentProj * ma1;

	    // look for sanity, should be ~0
	    std::cerr << "Dot last: " << lastProj * mp2lastA << std::endl;
	    std::cerr << "Dot current: " << currentProj * mp2lastA << std::endl;

	    osg::Matrix turn;
	    turn.makeRotate(currentProj, lastProj);*/
	    /*osg::Vec3 base = currentAPoint - lastAPoint;
	    base.normalize();

	    osg::Vec3 perp = mp2lastA ^ base;
	    perp.normalize();

	    if(perp.length() < 0.9)
	    {
		//std::cerr << "Vec not valid" << std::endl;
		return;
	    }

	    osg::Vec3 perpL = perp * osg::Matrix::rotate(_lastActiveHandMat.getRotate().inverse());
	    osg::Vec3 perpC = perp * osg::Matrix::rotate(_currentActiveHandMat.getRotate().inverse());

	    std::cerr << "perp x: " << perp.x() << " y: " << perp.y() << " z: " << perp.z() << std::endl;
	    std::cerr << "perpL x: " << perpL.x() << " y: " << perpL.y() << " z: " << perpL.z() << std::endl;
	    std::cerr << "perpC x: " << perpC.x() << " y: " << perpC.y() << " z: " << perpC.z() << std::endl;

	    osg::Matrix turn;
	    turn.makeRotate(perpL, perpC);*/

	    osg::Matrix mat = PluginHelper::getObjectMatrix();
	    mat = mat * osg::Matrix::translate(-midpoint) * ma1 * ma2 * osg::Matrix::translate(midpoint);
	    //mat = mat * osg::Matrix::translate(-midpoint) * turn * osg::Matrix::translate(midpoint);
	    PluginHelper::setObjectMatrix(mat);

	    break;
	}
	case SCALE:
	{
	    osg::Vec3 pos1 = _currentActiveHandMat.getTrans();
	    osg::Vec3 pos2 = _currentRefHandMat.getTrans();
	    float diff = (pos1 - pos2).length();
	    float newScale = (diff / _startScaleLength) * _startScale;
	    SceneManager::instance()->setObjectScale(newScale);
            std::cerr << "New Scale is: " << newScale << std::endl;
            osg::Matrix objmat = PluginHelper::getObjectTransform()->getMatrix();
	    osg::Vec3 offset = -((_scalePoint * newScale * objmat) - (_scalePoint * _startScale * objmat));
	    osg::Matrix m;
	    m.makeTranslate(offset);
	    m = _startXForm * m;
	    PluginHelper::setObjectMatrix(m);
	    break;
	}
	case NONE:
	default:
	    return;
    }
}
