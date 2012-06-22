#include "OsgEarth.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/PluginManager.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/ComController.h>
#include <cvrKernel/Navigation.h>
#include <cvrKernel/InteractionManager.h>
#include <cvrMenu/MenuSystem.h>
#include <PluginMessageType.h>
#include <iostream>

#include <osg/Matrix>
#include <osgDB/ReadFile>
#include <osg/Geode>
#include <osg/ShapeDrawable>
#include <osgUtil/IntersectVisitor>
#include <osgEarthDrivers/kml/KML>
#include <osg/Referenced>
#include <osg/PositionAttitudeTransform>

using namespace osg;
using namespace std;
using namespace cvr;
using namespace osgEarth;
using namespace osgEarth::Drivers;

const double earthRadiusMM = osg::WGS_84_RADIUS_EQUATOR * 1000.0;

CVRPLUGIN(OsgEarth)

OsgEarth::OsgEarth()
{

}

void OsgEarth::message(int type, char * data)
{
    // data needs to include the plugin name and also the lat,lon and height
    if(type == OE_ADD_MODEL)
    {
	// data contains 3 floats
	OsgEarthRequest request = * (OsgEarthRequest*) data;

	// if get a request create new node add matrix and return the address of the matrixtransform
        osg::Matrixd output;
	_map->getProfile()->getSRS()->getEllipsoid()->computeLocalToWorldTransformFromLatLongHeight(
	        osg::DegreesToRadians(request.lat),
	        osg::DegreesToRadians(request.lon),
		request.height,
	        output );

	if( request.trans != NULL )
	{
            request.trans->setMatrix(output);
	}
	else
	{
	    osg::MatrixTransform* trans = new osg::MatrixTransform();
            trans->setMatrix(output);
            SceneManager::instance()->getObjectsRoot()->addChild(trans);
            request.trans = trans;
	}

        // send message back	
	PluginManager::instance()->sendMessageByName(request.pluginName,OE_TRANSFORM_POINTER,(char *) &request);
    }
}

bool OsgEarth::init()
{
    std::cerr << "OsgEarth init\n";

    string baseEarth = ConfigManager::getEntry("Plugin.OsgEarth.Earth");

    osg::Node* earth = osgDB::readNodeFile(baseEarth);

    if( !earth )
    {
	cerr << "Error: No earth file added in config file under Plugin.OsgEarth.Earth" << endl;
 	return false;
    }

    // disable special culling of the planet and intersection
    earth->setNodeMask(earth->getNodeMask() & ~DISABLE_FIRST_CULL & ~INTERSECT_MASK);

    SceneManager::instance()->getObjectsRoot()->addChild(earth);

    // get the map to use for elevation
    _mapNode = MapNode::findMapNode( earth );
    _mapNode->setNodeMask(~2);
    _map = _mapNode->getMap();

    // set planet to correct scale
    osg::Matrix objects = PluginHelper::getObjectMatrix();
    objects.setTrans(0.0, earthRadiusMM * 3.0, 0.0);
    PluginHelper::setObjectScale(1000.0);
    PluginHelper::setObjectMatrix(objects);    // moves the planet; matrix should be a pure translation and rotation

    // add menu for flying nav
    _osgEarthMenu = new SubMenu("OsgEarth");

    // enable and disable visibility
    _visCB = new MenuCheckbox("Visible", true);
    _visCB->setCallback(this);
    _osgEarthMenu->addItem(_visCB);

    // enable and disable navigation
    _navCB = new MenuCheckbox("Planet Nav Mode", true);
    _navCB->setCallback(this);
    _osgEarthMenu->addItem(_navCB);

    PluginHelper::addRootMenuItem(_osgEarthMenu);

    _navActive = false;
    _mouseNavActive = false;

    return true;
}

void OsgEarth::preFrame()
{
    if(_navActive || _mouseNavActive)
    {
        double distanceToSurface = 0.0;

        // make sure values are the same across the tiles
        if(ComController::instance()->isMaster())
        {
            // need to get origin of cave in planet space (world origin is 0, 0, 0)
            osg::Vec3d origPlanetPoint = PluginHelper::getWorldToObjectTransform().getTrans();

            // planetPoint in latlonheight
            osg::Vec3d latLonHeight;
	    _map->getProfile()->getSRS()->getEllipsoid()->convertXYZToLatLongHeight(
				    origPlanetPoint.x(),
				    origPlanetPoint.y(),
				    origPlanetPoint.z(),
		                    latLonHeight.x(),
				    latLonHeight.y(),
				    latLonHeight.z());

            // set height back to the surface level 
            latLonHeight[2] = 0.0;

            // adjust the height to the ellipsoid
	    _map->getProfile()->getSRS()->getEllipsoid()->convertLatLongHeightToXYZ(
		                    latLonHeight.x(),
				    latLonHeight.y(),
				    latLonHeight.z(),
				    origPlanetPoint.x(),
				    origPlanetPoint.y(),
				    origPlanetPoint.z());


            distanceToSurface = (origPlanetPoint * PluginHelper::getObjectToWorldTransform()).length();

            ComController::instance()->sendSlaves(&distanceToSurface,sizeof(double));
        }
        else
        {
            ComController::instance()->readMaster(&distanceToSurface,sizeof(double));
        }

        //std::cerr << "distance: " << distanceToSurface << std::endl;

        if(_navActive)
        {
            processNav(getSpeed(distanceToSurface));
        }
        else
        {
            processMouseNav(getSpeed(distanceToSurface));
        }
    }
}

bool OsgEarth::buttonEvent(int type, int button, int hand, const osg::Matrix & mat)
{
    //std::cerr << "Button event." << std::endl;
    if(!_navCB->getValue() || Navigation::instance()->getPrimaryButtonMode() == SCALE)
    {
        return false;
    }

    if(!_navActive && button == 0 && (type == BUTTON_DOWN || type == BUTTON_DOUBLE_CLICK))
    {
        _navHand = hand;
        _navHandMat = mat;
        _navActive = true;
        _mouseNavActive = false;
        return true;
    }
    else if(!_navActive)
    {
        return false;
    }

    if(hand != _navHand)
    {
        return false;
    }

    if(button == 0)
    {
        if(type == BUTTON_UP)
        {
            _navActive = false;
        }
        return true;
    }

    return false;
}

bool OsgEarth::processEvent(InteractionEvent * event)
{
    MouseInteractionEvent * mie = event->asMouseEvent();
    if(mie)
    {
        return mouseButtonEvent(mie->getInteraction(),mie->getButton(),mie->getX(),mie->getY(),mie->getTransform());
    }

    TrackedButtonInteractionEvent * tie = event->asTrackedButtonEvent();
    if(tie)
    {
        return buttonEvent(tie->getInteraction(),tie->getButton(),tie->getHand(),tie->getTransform());
    }

    return false;
}

bool OsgEarth::mouseButtonEvent (int type, int button, int x, int y, const osg::Matrix &mat)
{
    if(!_navCB->getValue() || Navigation::instance()->getPrimaryButtonMode() == SCALE)
    {
        return false;
    }

    if(_navActive)
    {
        return false;
    }

    if(!_mouseNavActive && button == 0 && (type == BUTTON_DOWN || type == BUTTON_DOUBLE_CLICK))
    {
        _startX = x;
        _startY = y;
        _currentX = x;
        _currentY = y;
        _mouseNavActive = true;

        if(Navigation::instance()->getPrimaryButtonMode() == MOVE_WORLD)
        {
            _movePointValid = false;
        }

        return true;
    }
    else if(!_mouseNavActive)
    {
        return false;
    }

    if(button == 0)
    {
        if(type == BUTTON_UP)
        {
            _mouseNavActive = false;
        }
        _currentX = x;
        _currentY = y;
        return true;
    }

    return false;
}

void OsgEarth::menuCallback(MenuItem * item)
{
    if(item == _navCB)
    {
        _navActive = _navCB->getValue();
    }
    else if(item == _visCB)
    {
	if(_visCB->getValue())
		_mapNode->setNodeMask(~2);
	else
		_mapNode->setNodeMask(0);

    }
}

void OsgEarth::processNav(double speed)
{
    double time = PluginHelper::getLastFrameDuration();
    float rangeValue = 500.0;

    switch(Navigation::instance()->getPrimaryButtonMode())
    {
        case WALK:
        case DRIVE:
        {
            osg::Vec3 trans;
            osg::Vec3 offset = PluginHelper::getHandMat(_navHand).getTrans() - _navHandMat.getTrans();
            trans = offset;
            trans.normalize();
            offset.z() = 0.0;

            float speedScale = fabs(offset.length() / rangeValue);

            trans = trans * (speedScale * speed * time * 1000.0);

            osg::Matrix r;
            r.makeRotate(_navHandMat.getRotate());
            osg::Vec3 pointInit = osg::Vec3(0, 1, 0);
            pointInit = pointInit * r;
            pointInit.z() = 0.0;

            r.makeRotate(PluginHelper::getHandMat(_navHand).getRotate());
            osg::Vec3 pointFinal = osg::Vec3(0, 1, 0);
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
                else if((pointInit ^ pointFinal).z() < 0)
                {
                    angle = -angle;
                }
                turn.makeRotate(-angle, osg::Vec3(0, 0, 1));
            }

            osg::Matrix objmat = PluginHelper::getObjectMatrix();

            osg::Vec3 origin = PluginHelper::getHandMat(_navHand).getTrans();

            objmat = objmat * osg::Matrix::translate(-origin) * turn * osg::Matrix::translate(origin - trans);
            PluginHelper::setObjectMatrix(objmat);

            break;
        }
	case FLY:
        {
            osg::Vec3 trans = PluginHelper::getHandMat(_navHand).getTrans() - _navHandMat.getTrans();

            float speedScale = fabs(trans.length() / rangeValue);
            trans.normalize();

            trans = trans * (speedScale * speed * time * 1000.0);

            osg::Matrix rotOffset = osg::Matrix::rotate(_navHandMat.getRotate().inverse())
                                 * osg::Matrix::rotate(PluginHelper::getHandMat(_navHand).getRotate());
            osg::Quat rot = rotOffset.getRotate();
            rot = rot.inverse();
            double angle;
            osg::Vec3 vec;
            rot.getRotate(angle, vec);
            rot.makeRotate(angle / 20.0, vec);
            rotOffset.makeRotate(rot);

            /*osg::Matrix r;
            r.makeRotate(_navHandMat.getRotate());
            osg::Vec3 pointInit = osg::Vec3(0, 1, 0);
            pointInit = pointInit * r;

            r.makeRotate(PluginHelper::getHandMat(_navHand).getRotate());
            osg::Vec3 pointFinal = osg::Vec3(0, 1, 0);
            pointFinal = pointFinal * r;

            osg::Quat turn;
            if(pointInit.length2() > 0 && pointFinal.length2() > 0)
            {
                pointInit.normalize();
                pointFinal.normalize();
                turn.makeRotate(pointInit,pointFinal);
                
                double angle;
                osg::Vec3 vec;
                turn.getRotate(angle,vec);
                turn.makeRotate(angle / 20.0, vec);
            }*/

            osg::Matrix objmat = PluginHelper::getObjectMatrix();

            osg::Vec3 origin = PluginHelper::getHandMat(_navHand).getTrans();

            objmat = objmat * osg::Matrix::translate(-origin) * rotOffset * osg::Matrix::translate(origin - trans);
            PluginHelper::setObjectMatrix(objmat);
            break;
        }
        default:
            break;
    }
}

void OsgEarth::processMouseNav(double speed)
{
    int masterScreen = CVRViewer::instance()->getActiveMasterScreen();
    if(masterScreen < 0)
    {
        return;
    }

    ScreenInfo * si = ScreenConfig::instance()->getMasterScreenInfo(masterScreen);
    if(!si)
    {
        return;
    }

    osg::Vec3d screenCenter = si->xyz;
    osg::Vec3d screenDir(0,1,0);
    screenDir = screenDir * si->transform;
    screenDir = screenDir - screenCenter;
    screenDir.normalize();

    osg::Vec3d planetPoint(0,0,0);
    planetPoint = planetPoint * PluginHelper::getObjectToWorldTransform();
    double planetDist = (screenCenter - planetPoint).length();

    switch(Navigation::instance()->getPrimaryButtonMode())
    {
        case MOVE_WORLD:
        {
            osg::Vec3d P1(0,0,0),P2(0,1000000,0);
            P1 = P1 * PluginHelper::getMouseMat();
            P2 = P2 * PluginHelper::getMouseMat();

            osg::Vec3d lineDir = P2 - P1;
            lineDir.normalize();

            osg::Vec3d c = planetPoint - P1;
            double ldotc = lineDir * c;

            double determ = ldotc * ldotc - c * c + earthRadiusMM * earthRadiusMM;
            if(determ < 0)
            {
                _movePointValid = false;
                break;
            }

            double d;

            if(determ == 0)
            {
                d = ldotc;
            }
            else
            {
                double d1,d2;
                d1 = ldotc + sqrt(determ);
                d2 = ldotc - sqrt(determ);
                if(d1 >= 0.0 && d1 < d2)
                {
                    d = d1;
                }
								else if(d2 >= 0.0)
                {
                    d = d2;
                }
                else // intersect with planet behind viewer
                {
                    _movePointValid = false;
                    break;
                }
            }
            osg::Vec3d movePoint = lineDir * d + P1;

            if(!_movePointValid)
            {
                _movePoint = movePoint;
                _movePointValid = true;
                break;
            }

            P1 = _movePoint - planetPoint;
            P2 = movePoint - planetPoint;
            P1.normalize();
            P2.normalize();

            osg::Matrix objMat = PluginHelper::getObjectMatrix();
            objMat = objMat * osg::Matrix::translate(-planetPoint) * osg::Matrix::rotate(P1,P2) * osg::Matrix::translate(planetPoint);
            PluginHelper::setObjectMatrix(objMat);

            _movePoint = movePoint;
            osg::Matrix m;
            m.makeTranslate(_movePoint);
            //_testMark->setMatrix(m);

            break;
        }
        case WALK:
        case DRIVE:
        {
            osg::Vec3d planetDir = planetPoint - screenCenter;
            planetDir.normalize();

            double yDiff = _currentY - _startY;
            osg::Vec3d planetOffset = screenDir * yDiff * speed * 0.3;

            osg::Vec3 screen2Planet = screenDir * planetDist;

            osg::Vec3d screenUp(0,0,1);
            screenUp = screenUp * si->transform;
            screenUp = screenUp - screenCenter;
            screenUp.normalize();

            /*double dist = _currentX - _startX;
            dist /= si->myChannel->width;
            dist *= si->width;
            osg::Vec3d rotPoint(dist,0,0);
            rotPoint = rotPoint * si->transfrom;
            rotPoint = rotPoint - planetPoint;
            rotPoint.normalize();

            osg::Vec3d cpoint = -screen2Planet;
						cpoint.normalize();*/


            double angle = _currentX - _startX;
            //angle /= -100000.0;
            angle *= (earthRadiusMM - planetDist) / 500000000000000.0;
            //angle *= -speed / 200000000.0;


            osg::Matrix objMat = PluginHelper::getObjectMatrix();
            objMat = objMat * osg::Matrix::translate(-screenCenter) * osg::Matrix::rotate(planetDir,screenDir) * osg::Matrix::translate(-screen2Planet) * osg::Matrix::rotate(angle,screenUp) * osg::Matrix::translate(screen2Planet + planetOffset + screenCenter);
            PluginHelper::setObjectMatrix(objMat);
            break;
        }
        case FLY:
        {
            osg::Vec3d planetDir = planetPoint - screenCenter;
            planetDir.normalize();

            osg::Vec3d axis(_currentX - _startX, 0, -_currentY + _startY);
            double angle = axis.length();
            axis.normalize();

            axis = axis * si->transform;
            axis = axis ^ screenDir;
            axis.normalize();

            // check if invalid
            if(axis.length() < 0.9)
            {
                break;
            }

            osg::Vec3 screen2Planet = screenDir * planetDist;

            //angle *= -speed / 200000000.0;
            angle *= (earthRadiusMM - planetDist) / 200000000000000.0;

            osg::Matrix objMat = PluginHelper::getObjectMatrix();
            objMat = objMat * osg::Matrix::translate(-screenCenter) * osg::Matrix::rotate(planetDir,screenDir) * osg::Matrix::translate(-screen2Planet) * osg::Matrix::rotate(angle,axis) * osg::Matrix::translate(screen2Planet + screenCenter);
            PluginHelper::setObjectMatrix(objMat);

            break;
        }
        defaut:
            break;
    }
}

double OsgEarth::getSpeed(double distance)
{
    double boundDist = std::max((double)0.0,distance);
    boundDist /= 1000.0;

    if(boundDist < 762.0)
    {
        return 10.0 * (0.000000098 * pow(boundDist,3) + 1.38888);
    }
    else if(boundDist < 10000.0)
    {
        boundDist = boundDist - 762.0;
        return 10.0 * (0.0314544 * boundDist + 44.704);
    }
    else
    {
        boundDist = boundDist - 10000;
        double cap = 0.07 * boundDist + 335.28;
        cap = std::min((double)50000,cap);
        return 10.0 * cap;
    }
}

/*void OsgEarth::processMouseNav(double speed)
{
    int masterScreen = CVRViewer::instance()->getActiveMasterScreen();
    if(masterScreen < 0)
    {
        return;
    }

    ScreenInfo * si = ScreenConfig::instance()->getMasterScreenInfo(masterScreen);
    if(!si)
    {
        return;
    }

    osg::Vec3d screenCenter = si->xyz;
    osg::Vec3d screenDir(0,1,0);
    screenDir = screenDir * si->transform;
    screenDir = screenDir - screenCenter;
    screenDir.normalize();

    osg::Vec3d planetPoint(0,0,0);
    planetPoint = planetPoint * PluginHelper::getObjectToWorldTransform();
    double planetDist = (screenCenter - planetPoint).length();

    switch(Navigation::instance()->getPrimaryButtonMode())
    {
        case MOVE_WORLD:
        {
            osg::Vec3d P1(0,0,0),P2(0,1000000,0);
            P1 = P1 * PluginHelper::getMouseMat();
            P2 = P2 * PluginHelper::getMouseMat();

            osg::Vec3d lineDir = P2 - P1;
            lineDir.normalize();

            osg::Vec3d c = planetPoint - P1;
            double ldotc = lineDir * c;

            double determ = ldotc * ldotc - c * c + earthRadiusMM * earthRadiusMM;
            if(determ < 0)
            {
                _movePointValid = false;
                break;
            }

            double d;

            if(determ == 0)
	    {
                d = ldotc;
            }
            else
            {
                double d1,d2;
                d1 = ldotc + sqrt(determ);
                d2 = ldotc - sqrt(determ);
                if(d1 >= 0.0 && d1 < d2)
                {
                    d = d1;
                }
                                                                else if(d2 >= 0.0)
                {
                    d = d2;
                }
                else // intersect with planet behind viewer
                {
                    _movePointValid = false;
                    break;
                }
            }
            osg::Vec3d movePoint = lineDir * d + P1;

            if(!_movePointValid)
            {
                _movePoint = movePoint;
                _movePointValid = true;
                break;
            }

            P1 = _movePoint - planetPoint;
            P2 = movePoint - planetPoint;
            P1.normalize();
            P2.normalize();

            osg::Matrix objMat = PluginHelper::getObjectMatrix();
            objMat = objMat * osg::Matrix::translate(-planetPoint) * osg::Matrix::rotate(P1,P2) * osg::Matrix::translate(planetPoint);
            PluginHelper::setObjectMatrix(objMat);

            _movePoint = movePoint;
            osg::Matrix m;
            m.makeTranslate(_movePoint);
            //_testMark->setMatrix(m);

            break;
        }
        case WALK:
	case DRIVE:
        {
            osg::Vec3d planetDir = planetPoint - screenCenter;
            planetDir.normalize();

            double yDiff = _currentY - _startY;
            osg::Vec3d planetOffset = screenDir * yDiff * speed * 0.3;

            osg::Vec3 screen2Planet = screenDir * planetDist;

            osg::Vec3d screenUp(0,0,1);
            screenUp = screenUp * si->transform;
            screenUp = screenUp - screenCenter;
            screenUp.normalize();

            /*double dist = _currentX - _startX;
            dist /= si->myChannel->width;
            dist *= si->width;
            osg::Vec3d rotPoint(dist,0,0);
            rotPoint = rotPoint * si->transfrom;
            rotPoint = rotPoint - planetPoint;
            rotPoint.normalize();

            osg::Vec3d cpoint = -screen2Planet;
                                                cpoint.normalize();*/
/*

            double angle = _currentX - _startX;
            //angle /= -100000.0;
            angle *= (earthRadiusMM - planetDist) / 500000000000000.0;
            //angle *= -speed / 200000000.0;


            osg::Matrix objMat = PluginHelper::getObjectMatrix();
            objMat = objMat * osg::Matrix::translate(-screenCenter) * osg::Matrix::rotate(planetDir,screenDir) * osg::Matrix::translate(-screen2Planet) * osg::Matrix::rotate(angle,screenUp) * osg::Matrix::translate(screen2Planet + planetOffset + screenCenter);
            PluginHelper::setObjectMatrix(objMat);
            break;
        }
        case FLY:
        {
            osg::Vec3d planetDir = planetPoint - screenCenter;
            planetDir.normalize();

            osg::Vec3d axis(_currentX - _startX, 0, -_currentY + _startY);
            double angle = axis.length();
            axis.normalize();

            axis = axis * si->transform;
            axis = axis ^ screenDir;
	    axis.normalize();

            // check if invalid
            if(axis.length() < 0.9)
            {
                break;
            }

            osg::Vec3 screen2Planet = screenDir * planetDist;

            //angle *= -speed / 200000000.0;
            angle *= (earthRadiusMM - planetDist) / 200000000000000.0;

            osg::Matrix objMat = PluginHelper::getObjectMatrix();
            objMat = objMat * osg::Matrix::translate(-screenCenter) * osg::Matrix::rotate(planetDir,screenDir) * osg::Matrix::translate(-screen2Planet) * osg::Matrix::rotate(angle,axis) * osg::Matrix::translate(screen2Planet + screenCenter);
            PluginHelper::setObjectMatrix(objMat);

            break;
        }
        defaut:
            break;
    }
}

double OsgEarth::getSpeed(double distance)
{
    double boundDist = std::max((double)0.0,distance);
    boundDist/=1000.0;

    if(boundDist < 762.0)
    {
        return 10.0 * (0.000000098 * pow(boundDist,3) + 1.38888);
    }
    else if(boundDist < 10000.0)
    {
        boundDist = boundDist - 762.0;
        return 10.0 * (0.0314544 * boundDist + 44.704);
    }
    else
    {
        boundDist = boundDist - 10000;
        double cap = 0.07 * boundDist + 335.28;
        cap = std::min((double)50000,cap);
        return 10.0 * cap;
    }
}*/


OsgEarth::~OsgEarth()
{
}
