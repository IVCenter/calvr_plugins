#include "OssimPlanet.h"

#include <kernel/Navigation.h>
#include <kernel/InteractionManager.h>

#include <cmath>

using namespace cvr;
using namespace osg;

ossimPlanet* OssimPlanet::planet;
OssimPlanet* OssimPlanet::oplanet = NULL;

const double earthRadiusMM = osg::WGS_84_RADIUS_EQUATOR * 1000.0;
static const string FILES("Plugin.OssimPlanet.Kml");

// Constructor
OssimPlanet::OssimPlanet()
{

}

osg::Vec3d OssimPlanet::convertUtm84ToLatLon(int zone, char hemi, double northing, double easting)
{
     // translate to lon lat
     ossimGpt result;
     //ossimUtmpt test(36, 'N', 733463.783, 3396594.585, ossimDatumFactory::instance()->wgs84());
     ossimUtmpt point(zone, hemi, northing, easting, ossimDatumFactory::instance()->wgs84());
     point.convertToGround(result);
     return osg::Vec3d(result.lond(), result.latd(), result.heightMSL());
}

bool OssimPlanet::addModel(osg::Node* node, double lat, double lon, osg::Vec3 scale, double height, double heading, double pitch, double roll)
{
   // will only execute if the plugin has been loaded
   if(node == NULL || planet == NULL)
       return false;

   osg::Vec3d llh(lat,lon,height);
   llh[2] += planet->model()->getGeoidOffset(lat,
                                          lon);
   osg::Matrixd localToWorld;
   planet->model()->lsrMatrix(llh, localToWorld);

   // add similar as a kml file
   osg::Vec3d tempLlh(lat,
                      lon,
                      height + planet->model()->getHeightAboveEllipsoid(lat,
                                                   lon));

   osg::MatrixTransform* localToWorldTransform = new osg::MatrixTransform();

   planet->model()->lsrMatrix(tempLlh, localToWorld, heading);
   localToWorld = (osg::Matrixd::scale(scale)*
                   osg::Matrixd::rotate(pitch, osg::Vec3d(1.0, 0.0, 0.0))*
                   osg::Matrixd::rotate(roll, osg::Vec3d(0.0, 1.0, 0.0)))*localToWorld;
   localToWorldTransform->setMatrix(localToWorld);

   osg::MatrixTransform* modelTransform = new osg::MatrixTransform;
   double scale2 = (1.0/planet->model()->getNormalizationScale());
   modelTransform->setMatrix(osg::Matrixd::scale((scale2),
                                                 (scale2),
                                                 (scale2)));
   modelTransform->addChild(node);
   modelTransform->getOrCreateStateSet()->setMode(GL_RESCALE_NORMAL,osg::StateAttribute::ON);

   localToWorldTransform->addChild(modelTransform);
   ossimSetNonPowerOfTwoTextureVisitor nv;
   localToWorldTransform->accept(nv);
   planet->addChild(localToWorldTransform);

   return true;
}

OssimPlanet* OssimPlanet::instance()
{
    return OssimPlanet::oplanet;
}

// intialize
bool OssimPlanet::init()
{

   SceneManager::instance()->setDepthPartitionActive(true);

   // static setting
   if(!OssimPlanet::oplanet)
       oplanet = new OssimPlanet();

   ossimInit::instance()->initialize();
   //ossimPreferences::instance()->loadPreferences(ossimFilename("/home/covise/.ossim/ossim_preferences"));
   osg::ref_ptr<ossimPlanetGrid> grid = new ossimPlanetAdjustableCubeGrid(ossimPlanetAdjustableCubeGrid::MEDIUM_CAP);

   ossimKeywordlist kwl;
   kwl.addFile(ConfigManager::getEntry("Plugin.OssimPlanet.ConfigFile").c_str());
   osg::ref_ptr<ossimPlanetTextureLayer> groupLayer = ossimPlanetTextureLayerRegistry::instance()->createLayer(kwl.toString());

   ossimPlanetTerrainGeometryTechnique* technique = new ossimPlanetTerrainGeometryTechnique();

   planet = new ossimPlanet();

   ossimPlanetTerrain::CullAmountType cullAmount = ossimPlanetTerrain::HIGH_CULL;
   ossimPlanetTerrain::SplitMergeSpeedType splitMergeSpeed = ossimPlanetTerrain::MEDIUM_SPEED;
   ossimPlanetTerrain::ElevationDensityType elevationDensity = ossimPlanetTerrain::MEDIUM_ELEVATION_DENSITY;
   ossimPlanetTerrain::TextureDensityType textureDensity = ossimPlanetTerrain::MEDIUM_TEXTURE_DENSITY;
   double minTimeToCompilePerFrame = .003;

   osg::ref_ptr<ossimPlanetTerrain> terrain = new ossimPlanetTerrain(grid.get());
   terrain->setPrecompileEnabledFlag(false);
   terrain->setTerrainTechnique(technique);
   terrain->setCullAmountType(cullAmount);
   terrain->setSplitMergeSpeedType(splitMergeSpeed);
   terrain->setTextureDensityType(textureDensity);
   terrain->setElevationDensityType(elevationDensity);
   terrain->setElevationExaggeration(1.0);
   terrain->setMinimumTimeToCompilePerFrameInSeconds(minTimeToCompilePerFrame);
   terrain->setElevationMemoryCache(new ossimPlanetMemoryImageCache);
   terrain->elevationCache()->setMinMaxCacheSizeInMegaBytes(128, 256);

   // add texture  
   if( groupLayer.valid() )
   {
      terrain->setTextureLayer(0, groupLayer.get());
   }

   terrain->initElevation();
   terrain->setElevationEnabledFlag(true);

   std::vector<ossimFilename> kmlFiles;
   osg::ref_ptr<ossimPlanetKmlLayer> kmlLayer;


   // read in kml from configuration file
   std::vector<std::string> tagList;
   ConfigManager::getChildren(FILES, tagList);

   for(int i = 0; i < tagList.size(); i++)
   {
        std::string tag = FILES + "." + tagList[i];
	kmlFiles.push_back(ossimFilename(ConfigManager::getEntry("value", tag, "")));
   }

   if(kmlFiles.size() > 0)
   {
      kmlLayer = new ossimPlanetKmlLayer();
      planet->addChild(kmlLayer.get());
      ossim_uint32 idx = 0;
      for(idx = 0; idx < kmlFiles.size(); ++idx)
      {
         kmlLayer->addKml(kmlFiles[idx]);
      }
   }
   
      
   // set ephemeris
   ephemeris = NULL;
   
   //set object transform and ignore root transform
   osg::Matrix objects = PluginHelper::getObjectMatrix();
   objects.setTrans(0.0, earthRadiusMM * 3.0, 0.0);  // scales the planet
   PluginHelper::setObjectMatrix(objects);    // moves the planet; matrix should be a pure translation and rotation
   PluginHelper::setObjectScale(earthRadiusMM); // this matrix should be a pure scale
   
   planet->addChild(terrain.get());
   
   // enable cloud cover
   cloudCover();

   //osg::Group *models = new osg::Group();
   //readArtifactsFile(ConfigManager::getEntry("Plugin.ArtifactVis.DataBase"));
   //displayArtifacts(models);
   //addModel(models, 30.628039, 35.491239, osg::Vec3(0.06, 0.06, 0.06), 20.0, 0.0, 90.0, 135.03); 
   //addModel(models, 30.628039, 35.491239, osg::Vec3(0.06, 0.06, 0.06), 10.0, 0.0, 90.0, 135.0); 

   _ossimMenu = new SubMenu("OssimPlanet");

   _navCB = new MenuCheckbox("Planet Nav Mode",true);
   _navCB->setCallback(this);
   _ossimMenu->addItem(_navCB);
   PluginHelper::addRootMenuItem(_ossimMenu);

   _navActive = false;
   _mouseNavActive = false;

   SceneManager::instance()->getObjectsRoot()->addChild(planet);

    /*Material * mat =new Material();	
    mat->setColorMode(Material::AMBIENT_AND_DIFFUSE);
    mat->setDiffuse(Material::FRONT,osg::Vec4(1.0,1.0,1.0,1.0));
    
    osg::Sphere * ssph = new osg::Sphere(osg::Vec3(0,0,0),0.00001);
    osg::ShapeDrawable * sd = new osg::ShapeDrawable(ssph);
    sd->setColor(osg::Vec4(1.0,0,0,1.0));
    osg::StateSet * stateset = sd->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING, osg::StateAttribute::ON);
    stateset->setAttributeAndModes(mat,osg::StateAttribute::ON);

    _testMark = new osg::MatrixTransform();
    osg::Geode * geo = new osg::Geode();
    geo->addDrawable(sd);
    _testMark->addChild(geo);

    PluginHelper::getObjectsRoot()->addChild(_testMark);*/

   return true;
}

void OssimPlanet::getObjectIntersection(osg::Node *root, osg::Vec3& wPointerStart, osg::Vec3& wPointerEnd, IsectInfo& isect)
{
        // Compute intersections of viewing ray with objects:
	osgUtil::IntersectVisitor iv;
	osg::ref_ptr<osg::LineSegment> testSegment = new osg::LineSegment();
	testSegment->set(wPointerStart, wPointerEnd);
	iv.addLineSegment(testSegment.get());
	iv.setTraversalMask(2);
	
	// Traverse the whole scenegraph.
	// Non-Interactive objects must have been marked with setNodeMask(~2):     
	root->accept(iv);
	isect.found = false;
	if (iv.hits())
	{
	    osgUtil::IntersectVisitor::HitList& hitList = iv.getHitList(testSegment.get());
	    if(!hitList.empty())
	    {
	         isect.point     = hitList.front().getWorldIntersectPoint();
	         isect.normal    = hitList.front().getWorldIntersectNormal();
	         isect.geode     = hitList.front()._geode.get();
	         isect.found     = true;

		 //osg::Vec3d llh;
		 //osg::Vec3d wpt = isect.point * PluginHelper::getWorldToObjectTransform();
                 //planet->model()->inverse(wpt, llh);
		 //printf("Long lat and height is %f %f %f\n", llh.x(), llh.y(), llh.z());
	    }
	}
}

void OssimPlanet::cloudCover()
{

   osg::Camera* theEphemerisCamera = NULL;
 
   ossimFilename sunTextureFile = "";
   ossimFilename moonTextureFile = "";
   //ossim_float64 visibility = 1000000000.0;
   ossim_float64 visibility = 1000.0;
   ossim_float64 fogNear = 10.0;
   ossim_int32 cloudCoverage = 20;
   ossim_float64 cloudSharpness = .95;
   ossim_float64 cloudAltitude = 20000;

   ossimLocalTm date;
   date.setYear(2011);
   date.setMonth(1);
   date.setDay(1);
   date.setHour(3);
   //date.now();

   ossimPlanet* tempPlanet = new ossimPlanet;
   tempPlanet->setComputeIntersectionFlag(false);

   ephemeris = new ossimPlanetEphemeris(ossimPlanetEphemeris::MOON_LIGHT | ossimPlanetEphemeris::SKY | ossimPlanetEphemeris::AMBIENT_LIGHT | ossimPlanetEphemeris::SUN_LIGHT | ossimPlanetEphemeris::FOG);
   theEphemerisCamera = new osg::Camera;

   tempPlanet->addChild(ephemeris);
   SceneManager::instance()->getObjectsRoot()->addChild(tempPlanet);

   ephemeris->setRoot(SceneManager::instance()->getObjectsRoot());

   theEphemerisCamera->setProjectionResizePolicy(CVRViewer::instance()->getCamera()->getProjectionResizePolicy());
   theEphemerisCamera->setClearColor(CVRViewer::instance()->getCamera()->getClearColor());
   theEphemerisCamera->setRenderOrder(osg::Camera::PRE_RENDER);
   theEphemerisCamera->setRenderTargetImplementation( CVRViewer::instance()->getCamera()->getRenderTargetImplementation() );
   theEphemerisCamera->setClearMask(GL_COLOR_BUFFER_BIT);

   CVRViewer::instance()->getCamera()->setClearMask(CVRViewer::instance()->getCamera()->getClearMask() & ~GL_COLOR_BUFFER_BIT);

   if(CVRViewer::instance()->getCamera()->getViewport())
   {
         theEphemerisCamera->setViewport(new osg::Viewport(*CVRViewer::instance()->getCamera()->getViewport()));
   }
   else
   {
         theEphemerisCamera->setViewport(new osg::Viewport());
   }

   osg::Matrix theCurrentViewMatrix = CVRViewer::instance()->getCamera()->getViewMatrix() * PluginHelper::getWorldToObjectTransform();
   osg::Matrix theCurrentViewMatrixInverse = theCurrentViewMatrix.inverse(theCurrentViewMatrix);
   theEphemerisCamera->setViewMatrix(theCurrentViewMatrixInverse);

   CVRViewer::instance()->addSlave(theEphemerisCamera, false);
   ephemeris->setCamera(theEphemerisCamera);
   //theEphemerisCamera->setEventCallback(new ossimPlanetTraverseCallback());
   //theEphemerisCamera->setUpdateCallback(new ossimPlanetTraverseCallback());
   //theEphemerisCamera->setCullCallback(new ossimPlanetTraverseCallback());

   ephemeris->setDate(date);
   ephemeris->setApplySimulationTimeOffsetFlag(true);
   ephemeris->setSunTextureFromFile(sunTextureFile);
   ephemeris->setMoonTextureFromFile(moonTextureFile);
   ephemeris->setMoonScale(osg::Vec3d(1.0, 1.0, 1.0));
   ephemeris->setGlobalAmbientLight(osg::Vec3d(0.1, 0.1, 0.1));
   ephemeris->setVisibility(visibility);
   ephemeris->setFogNear(fogNear);
   ephemeris->setFogMode(ossimPlanetEphemeris::LINEAR);

   ephemeris->setNumberOfCloudLayers(1);
   ephemeris->cloudLayer(0)->computeMesh(cloudAltitude, 128, 128, 0);
   ephemeris->cloudLayer(0)->updateTexture(0, cloudCoverage, cloudSharpness);
   //ephemeris->cloudLayer(0)->setSpeedPerHour(1000, OSSIM_MILES);
   ephemeris->cloudLayer(0)->setSpeedPerHour(10000, OSSIM_MILES);
   ephemeris->cloudLayer(0)->setScale(3);
   ephemeris->cloudLayer(0)->setMaxAltitudeToShowClouds(cloudAltitude*2.0);
}

// this is called if the plugin is removed at runtime
OssimPlanet::~OssimPlanet()
{
   fprintf(stderr,"OssimPlanet::~OssimPlanet\n");
}


void OssimPlanet::preFrame()
{
/*
    osg::Matrix theCurrentViewMatrix = CVRViewer::instance()->getCamera()->getViewMatrix() * PluginHelper::getWorldToObjectTransform() * planetRescale;
    osg::Matrix theCurrentViewMatrixInverse = theCurrentViewMatrix.inverse(theCurrentViewMatrix);
    theEphemerisCamera->setViewMatrix(theCurrentViewMatrixInverse);
*/

    /*osg::Matrix originToPlanetObjectSpace = PluginHelper::getWorldToObjectTransform();
    // distance to surface in planet units (default without using intersection with surface)
    double distanceToSurface = originToPlanetObjectSpace.getTrans().length() - 1.0;
    
    IsectInfo itest;
    osg::Vec3 originObject(0.0,0.0,0.0);
    osg::Vec3 endObject = PluginHelper::getObjectToWorldTransform().getTrans();
    getObjectIntersection(SceneManager::instance()->getScene(), originObject, endObject, itest);

    if(itest.found)
    {
	if(ComController::instance()->isMaster())
	{
	    distanceToSurface = (originToPlanetObjectSpace.getTrans() - (itest.point * PluginHelper::getWorldToObjectTransform())).length(); 
	    ComController::instance()->sendSlaves(&distanceToSurface,sizeof(double));
	}
	else
	{
	    ComController::instance()->readMaster(&distanceToSurface,sizeof(double));
	}
    }*/

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
	    planet->model()->inverse(origPlanetPoint, latLonHeight);
  
	    // set height back to the surface level 
	    latLonHeight[2] = 0.0;

	    // adjust the height to the ellipsoid
	    planet->model()->mslToEllipsoidal(latLonHeight);

	    // translate point back to cartesian (in planet space)
	    osg::Vec3d pointObject;
	    planet->model()->forward(latLonHeight, pointObject);

	    distanceToSurface = (pointObject * PluginHelper::getObjectToWorldTransform()).length();

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

    /*double minNavScale = 20.0;
    double maxNavScale = 2000000.0;
    double minDistance = 0.0;   
    double maxDistance = 0.3;  //ratio is multiplied by the earth radius
    double ratio =  (distanceToSurface - minDistance) / (maxDistance - minDistance);

    if(distanceToSurface <= minDistance)
    {
        ratio = 0.0;
    }
    else if(distanceToSurface >= maxDistance)
    {
        ratio = 1.0;
    }

    double scaleFactor = minNavScale + (ratio * (maxNavScale - minNavScale));
    Navigation::instance()->setScale(scaleFactor);*/
}

bool OssimPlanet::buttonEvent(int type, int button, int hand, const osg::Matrix & mat)
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

bool OssimPlanet::mouseButtonEvent (int type, int button, int x, int y, const osg::Matrix &mat)
{
    if(!_navCB->getValue() || Navigation::instance()->getPrimaryButtonMode() == SCALE)
    {
	return false;
    }

    if(_navActive)
    {
	return false;
    }

    if(!_mouseNavActive && button == 0 && (type == MOUSE_BUTTON_DOWN || type == MOUSE_DOUBLE_CLICK))
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
	if(type == MOUSE_BUTTON_UP)
	{
	    _mouseNavActive = false;
	}
	_currentX = x;
	_currentY = y;
	return true;
    }

    return false;
}

void OssimPlanet::menuCallback(MenuItem * item)
{
    if(item == _navCB)
    {
	_navActive = false;
    }
}

void OssimPlanet::processNav(double speed)
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

void OssimPlanet::processMouseNav(double speed)
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

double OssimPlanet::getSpeed(double distance)
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

CVRPLUGIN(OssimPlanet)
