#include "OsgEarth.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/PluginManager.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/ComController.h>
#include <cvrKernel/Navigation.h>
#include <cvrKernel/InteractionManager.h>
#include <cvrKernel/NodeMask.h>
#include <cvrMenu/MenuSystem.h>
#include <PluginMessageType.h>
#include <iostream>

#include <osg/Matrix>
#include <osgDB/ReadFile>
#include <osg/Geode>
#include <osg/Image>
#include <osg/CullFace>
#include <osg/ShapeDrawable>
#include <osg/Referenced>
#include <osg/PositionAttitudeTransform>
#include <osgTerrain/TerrainTile>
#include <osgTerrain/GeometryTechnique>
#include <osgEarthDrivers/kml/KML>
#include <osg/Referenced>
#include <osg/PositionAttitudeTransform>
#include <osgEarth/TerrainOptions>
#include <osgEarth/MapNodeOptions>
#include <osgEarth/optional>

#include <osgDB/ReaderWriter>
#include <osgDB/Registry>


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

void OsgEarth::message(int type, char *&data, bool collaborative)
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
	    PluginManager::instance()->sendMessageByName(request.plugin,OE_TRANSFORM_POINTER,(char *) &request);
    }
    else if(type == OE_MENU)
    {
        OsgEarthMenuRequest request = * (OsgEarthMenuRequest*) data;
        request.oe_menu = _osgEarthMenu;
        PluginManager::instance()->sendMessageByName(request.plugin,OE_MENU,(char *) &request);
    } 
}

bool OsgEarth::init()
{
    std::cerr << "OsgEarth init\n";

    string baseEarth = ConfigManager::getEntry("Plugin.OsgEarth.Earth");

    osg::Node* earth = osgDB::readNodeFile(baseEarth);

    if(!earth)
    {
        std::cerr << "OsgEarth error: Unable to load earth: " << baseEarth << std::endl;
        return false;
    }

    // disable special culling of the planet
    earth->setNodeMask(earth->getNodeMask() & ~DISABLE_FIRST_CULL & ~INTERSECT_MASK);
    SceneManager::instance()->getObjectsRoot()->addChild(earth);

    // get the map to use for elevation
    _mapNode = MapNode::findMapNode( earth );
    _mapNode->setNodeMask(_mapNode->getNodeMask() & ~INTERSECT_MASK);
    _map = _mapNode->getMap();

    // loop through the configuration and add models to the planet
    // Load a KML file if specified
    vector<string> list;
    string configBase = "Plugin.OsgEarth.Models";
    ConfigManager::getChildren(configBase,list);

    for(int i = 0; i < list.size(); i++)
    {
            //KMLOptions kmlo;
            //kmlo.defaultIconImage() = URI("http://www.osgearth.org/chrome/site/pushpin_yellow.png").readImage();
            //osg::Node* kml = KML::load( URI(kmlFile), mapNode, kmlo );

            string file = ConfigManager::getEntry(configBase + "." + list[i]);
            printf("Trying to load %s\n", file.c_str());
            osg::Node* kml = KML::load( URI(file), _mapNode);
            
            if ( kml )
            {
                SceneManager::instance()->getObjectsRoot()->addChild( kml );
            }
    }

    // set planet to correct scale
    osg::Matrix objects = PluginHelper::getObjectMatrix();
    objects.setTrans(0.0, earthRadiusMM * 2.0, 0.0);
    PluginHelper::setObjectScale(1000.0);
    PluginHelper::setObjectMatrix(objects);    // moves the planet; matrix should be a pure translation and rotation

    // add menu for flying nav
    _osgEarthMenu = new SubMenu("OsgEarth");

    // enable and disable visibility
    _visCB = new MenuCheckbox("Visible", true);
    _visCB->setCallback(this);
    _osgEarthMenu->addItem(_visCB);

    // enable and disable navigation
    _navCB = new MenuCheckbox("Planet Nav Mode", false);
    _navCB->setCallback(this);
    _osgEarthMenu->addItem(_navCB);

    PluginHelper::addRootMenuItem(_osgEarthMenu);

    _navActive = false;
    _mouseNavActive = false;

    return true;
}

/*
void OsgEarth::setViewpoint( const Viewpoint& vp, double duration_s )
{
    if ( !established() ) 
    {
        _pending_viewpoint = vp;
        _pending_viewpoint_duration_s = duration_s;
        _has_pending_viewpoint = true;
    }

    else if ( duration_s > 0.0 )
    {
        // xform viewpoint into map SRS
        osg::Vec3d vpFocalPoint = vp.getFocalPoint();

        _start_viewpoint = getViewpoint(); //TODO
        
        _delta_heading = vp.getHeading() - _start_viewpoint.getHeading(); //TODO: adjust for crossing -180
        _delta_pitch   = vp.getPitch() - _start_viewpoint.getPitch();
        _delta_range   = vp.getRange() - _start_viewpoint.getRange();
        _delta_focal_point = vpFocalPoint - _start_viewpoint.getFocalPoint(); // TODO: adjust for lon=180 crossing

        while( _delta_heading > 180.0 ) _delta_heading -= 360.0;
        while( _delta_heading < -180.0 ) _delta_heading += 360.0;

        // adjust for geocentric date-line crossing
        while( _delta_focal_point.x() > 180.0 ) _delta_focal_point.x() -= 360.0;
        while( _delta_focal_point.x() < -180.0 ) _delta_focal_point.x() += 360.0;

        // calculate an acceleration factor based on the Z differential
        double h0 = _start_viewpoint.getRange() * sin( osg::DegreesToRadians(-_start_viewpoint.getPitch()) );
        double h1 = vp.getRange() * sin( osg::DegreesToRadians( -vp.getPitch() ) );
        double dh = (h1 - h0);

        // calculate the total distance the focal point will travel and derive an arc height:
        double de;
        
        osg::Vec3d startFP = _start_viewpoint.getFocalPoint();
        double x0,y0,z0, x1,y1,z1;
        _map->getProfile()->getSRS()->getEllipsoid()->convertLatLongHeightToXYZ(
            osg::DegreesToRadians( _start_viewpoint.y() ), osg::DegreesToRadians( _start_viewpoint.x() ), 0.0, x0, y0, z0 );
        _map->getProfile()->getSRS()->getEllipsoid()->convertLatLongHeightToXYZ(
            osg::DegreesToRadians( vpFocalPoint.y() ), osg::DegreesToRadians( vpFocalPoint.x() ), 0.0, x1, y1, z1 );
        de = (osg::Vec3d(x0,y0,z0) - osg::Vec3d(x1,y1,z1)).length();
        
                 
        _arc_height = osg::maximum( de - fabs(dh), 0.0 );

        // calculate acceleration coefficients
        if ( _arc_height > 0.0 )
        {
            // if we're arcing, we need separate coefficients for the up and down stages
            double h_apex = 2.0*(h0+h1) + _arc_height;
            double dh2_up = fabs(h_apex - h0)/100000.0;
            _set_viewpoint_accel = log10( dh2_up );
            double dh2_down = fabs(h_apex - h1)/100000.0;
            _set_viewpoint_accel_2 = -log10( dh2_down );
        }
        else
        {
            // on arc => simple unidirectional acceleration:
            double dh2 = (h1 - h0)/100000.0;
            _set_viewpoint_accel = fabs(dh2) <= 1.0? 0.0 : dh2 > 0.0? log10( dh2 ) : -log10( -dh2 );
            if ( fabs( _set_viewpoint_accel ) < 1.0 ) _set_viewpoint_accel = 0.0;
        }
        
        
        // don't use _time_s_now; that's the time of the last event
        _time_s_set_viewpoint = osg::Timer::instance()->time_s();
        _set_viewpoint_duration_s = duration_s;

        _setting_viewpoint = true;
        
        // recalculate the center point.
        recalculateCenter(); //TODO
    }
    else
    {
        osg::Vec3d new_center = vp.getFocalPoint();

        // start by transforming the requested focal point into world coordinates:
        if ( getSRS() )
        {
            // resolve the VP's srs. If the VP's SRS is not specified, assume that it
            // is either lat/long (if the map is geocentric) or X/Y (otherwise).
            osg::ref_ptr<const SpatialReference> vp_srs = vp.getSRS()? vp.getSRS() :
                _is_geocentric? getSRS()->getGeographicSRS() :
                getSRS();

    //TODO: streamline
            if ( !getSRS()->isEquivalentTo( vp_srs.get() ) )
            {
                osg::Vec3d local = new_center;
                // reproject the focal point if necessary:
                vp_srs->transform2D( new_center.x(), new_center.y(), getSRS(), local.x(), local.y() );
                new_center = local;
            }

            // convert to geocentric coords if necessary:
            if ( _is_geocentric )
            {
                osg::Vec3d geocentric;

                getSRS()->getEllipsoid()->convertLatLongHeightToXYZ(
                    osg::DegreesToRadians( new_center.y() ),
                    osg::DegreesToRadians( new_center.x() ),
                    new_center.z(),
                    geocentric.x(), geocentric.y(), geocentric.z() );

                new_center = geocentric;            
            }
        }

        // now calculate the new rotation matrix based on the angles:


        double new_pitch = osg::DegreesToRadians(
            osg::clampBetween( vp.getPitch(), _settings->getMinPitch(), _settings->getMaxPitch() ) );

        double new_azim = normalizeAzimRad( osg::DegreesToRadians( vp.getHeading() ) );

        setCenter( new_center ); // TODO
		setDistance( vp.getRange() ); // TODO

        _previousUp = getUpVector( _centerLocalToWorld ); //TODO

        _centerRotation = getRotation( new_center ).getRotate().inverse(); //TODO

		osg::Quat azim_q( new_azim, osg::Vec3d(0,0,1) );
        osg::Quat pitch_q( -new_pitch -osg::PI_2, osg::Vec3d(1,0,0) );

		osg::Matrix new_rot = osg::Matrixd( azim_q * pitch_q );

		_rotation = osg::Matrixd::inverse(new_rot).getRotate();

        recalculateCenter(); //TODO
    }
}
*/


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
    if(item == _visCB)
    {
	if(_visCB->getValue())
		_mapNode->setNodeMask(~2);
	else
		_mapNode->setNodeMask(0);

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

OsgEarth::~OsgEarth()
{
}
