#include "CameraFlight.h"
#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/InteractionManager.h>

#include <cvrKernel/ComController.h>
#include <cvrUtil/Intersection.h>
#include <cvrUtil/LocalToWorldVisitor.h>

#include <osgUtil/SceneView>
#include <osg/Camera>
#include <osgDB/ReadFile>
#include <math.h>
#include <osgAnimation/Sampler>

CVRPLUGIN(CameraFlight)

using namespace std;
using namespace osg;
using namespace cvr;
using namespace osgEarth;
osg::Vec3 trans1, scale1, trans, scale;
osg::Vec3d latLonHeight, origPlanetPoint;
osg::Vec3d fromVec, toVec;
osg::Quat rot1, so1, rot, so;
time_t tstart, tend;
osg::Matrix _origMatrix;
osg::Matrix _zoomMat;
osg::Matrix _rotMat;

osg::Vec3d crsV1, crsV2;

int rotdir;

double _origScale;
double origAngle, rotAngle, angle;
double planeOrigAngle, planeRotAngle, planeAngle;
double distanceToSurface = 0.0;

double maxHeight = 1.5e+10;
double minHeight = 6.5e+9;
double tmpHeight = 0;
double curHeight = 0;
double maxPlaneHeight = 1.5e+8;
double PIE = 3.14159265;
double curSurface = 0;
double a = 0.0;
		
double b = 0.0;
double t = 0.0;
double total = 0.0;

osg::Vec3 _destVec;
osg::Matrix objMat;
osg::Matrix o2w, w2o, rotMat;
bool flagZoom = false;
bool flagRot = false;
bool flagRise = false;
bool flagLower = false;
bool direction = false;
bool chk = false;
const double earthRadiusMM = osg::WGS_84_RADIUS_EQUATOR;

void CameraFlight::printMat(osg::Matrix m, double d)
{

    std::cerr<<"<<<<<<<<<<<<<<<<<<<<<<<<<<<<"<<endl;
    for (int i = 0; i < 4; i++) {
	for (int j = 0; j < 4; j++) {
	    std::cerr<<m(i,j)<<" ";
	}
	std::cerr<<std::endl;
    }
    std::cerr<<"Current Scale = "<<d<<std::endl;
    std::cerr<<"<<<<<<<<<<<<<<<<<<<<<<<<<<<<"<<endl;
}

void CameraFlight::printVec(osg::Vec3 v)
{
    for (int i = 0; i < 3; i++) {
	std::cerr<<v[i]<<" ";
    }
    std::cerr<<std::endl;
}

void CameraFlight::printQuat(osg::Quat q)
{

    for (int i = 0; i < 4; i++) {
	std::cerr<<q[i]<<" ";
    }
    std::cerr<<std::endl;
}

void CameraFlight::normalView() {

   objMat = SceneManager::instance()->getObjectTransform()->getMatrix();
   objMat.decompose(trans1, rot1, scale1, so1);

   trans1[2] = 0;
		

   objMat.setTrans(trans1);

   osg::Matrix rotM;
   rotM.makeRotate(DegreesToRadians(90.0),osg::Vec3(1,0,0));
   
   _rotMat = objMat * rotM;

   trans1[1] = tmpHeight;
   _rotMat.setTrans(trans1);
 
   SceneManager::instance()->setObjectMatrix(_rotMat);
}

void CameraFlight::planeView(){
  
   objMat = SceneManager::instance()->getObjectTransform()->getMatrix();
   objMat.decompose(trans1, rot1, scale1, so1);

   tmpHeight = trans1[1];

   trans1[1] = 0;

   objMat.setTrans(trans1);

   osg::Matrix rotM;
   rotM.makeRotate(DegreesToRadians(-90.0),osg::Vec3(1,0,0));
   
   _rotMat = objMat * rotM;

   trans1[2] = -(tmpHeight - distanceToSurface)-50000;
   _rotMat.setTrans(trans1);
   
   SceneManager::instance()->setObjectMatrix(_rotMat);
}

void CameraFlight::planeDir(osg::Vec3 fromV, osg::Vec3 toV) {
	
    objMat = SceneManager::instance()->getObjectTransform()->getMatrix();
    objMat.decompose(trans, rot, scale, so);

    osg::Vec3d xVector;

    crsV1 = fromV^toV;
    crsV1.normalize();

    osg::Matrix rotM;
    rotM.makeRotate(DegreesToRadians(-90.0), crsV1);

    _rotMat = rotM * objMat;

    SceneManager::instance()->setObjectMatrix(_rotMat);

    crsV1 = PluginHelper::getWorldToObjectTransform().getTrans();
    crsV1.normalize();

    SceneManager::instance()->setObjectMatrix(objMat);

    rotM.makeRotate(DegreesToRadians(90.0), osg::Vec3(1,0,0));
    objMat.setTrans(osg::Vec3(0,0,0));

    _rotMat = objMat * rotM;

    _rotMat.setTrans(trans);

    SceneManager::instance()->setObjectMatrix(_rotMat);

    crsV2 = PluginHelper::getWorldToObjectTransform().getTrans();
    crsV2.normalize();
    
    xVector = crsV1^crsV2;

    if(xVector.z() > 0){
	if((origPlanetPoint.z()) > 0)
	    rotdir = 1;
	else
	    rotdir = -1;
    }
    else{ 
	if((origPlanetPoint.z()) < 0)
	    rotdir = 1;
	else
	    rotdir = -1;
    }

    objMat.setTrans(trans);
    SceneManager::instance()->setObjectMatrix(objMat);
		
    planeOrigAngle = acos((crsV2 * crsV1)/((crsV2.length() * crsV1.length())));	
    planeOrigAngle = RadiansToDegrees(planeOrigAngle);

    planeAngle = planeOrigAngle;
    direction = true;

}

void CameraFlight::directSet(int dir){

  objMat = SceneManager::instance()->getObjectTransform()->getMatrix();
  objMat.decompose(trans1, rot1, scale1, so1);
  
  planeRotAngle = planeOrigAngle/100;

  if(planeAngle > 0.0) {
    osg::Matrix rotM;
    rotM.makeRotate(DegreesToRadians(dir*planeRotAngle),origPlanetPoint);

    planeAngle -= planeRotAngle;
    _rotMat = rotM * objMat;
  }

  else {
    direction = false;
    flagRot = true;
  }
  SceneManager::instance()->setObjectMatrix(_rotMat);
}

double CameraFlight::findMaxHeight(double theta) {

   objMat = SceneManager::instance()->getObjectTransform()->getMatrix();
   objMat.decompose(trans1, rot1, scale1, so1);

   if(theta > 45)
	theta = 45;
   double ret = ((maxPlaneHeight)*(theta/45.0));

   return ret;
}

/*MULTIPLY ROTANGLE TO DETERMINE HOW FAST YOU RISE*/
void CameraFlight::rise(osg::Vec3 v, osg::Matrix mat) {

   double heightdiff;

   if((origAngle - angle) <= (origAngle/4.0)) {
	v[2] = (maxHeight/2) * cos(PIE * (origAngle - angle)/(origAngle/4.0)) + (curHeight - (maxHeight/2));
   }


   else if (angle <= (origAngle/4.0)) {
	if(distanceToSurface < 50000)
	    flagRot = false;
	else
	    v[2] = (maxHeight/2) * cos(PIE + (PIE * (1.0 - (angle/(origAngle/4.0))))) + (-6.373e+09 - (maxHeight/2));
   }

   else {
   }

   mat.setTrans(v);
   SceneManager::instance()->setObjectMatrix(mat);
}

void CameraFlight::zoomOut(osg::Vec3 v, osg::Matrix mat)
{
    if(_flightMode != INSTANT) {
        if(t < 5.0) {
	    a = (maxHeight - trans[1])/25.0;
	    v[1] = (-a*pow((t-5),2)) + maxHeight;
        }

        else if (5.0 < t && t < 10.0){
	    a = (maxHeight - minHeight)/25.0;
	    v[1] = (-a*pow((t-5),2)) + maxHeight;
    	}

	else {
	    cout<<"Done"<<endl;
	    flagRot = false;
	}
    }   

    mat.setTrans(v);
    SceneManager::instance()->setObjectMatrix(mat);
}

void CameraFlight::rotate(osg::Vec3 from, osg::Vec3 to)
{
    objMat = SceneManager::instance()->getObjectTransform()->getMatrix();
    objMat.decompose(trans1, rot1, scale1, so1);

    if(_flightMode == INSTANT) {
	rotAngle = angle;
    }

    else if(_flightMode == AIRPLANE) {
	rotAngle = (origAngle + (-(origAngle/900)*pow((t-30),2)))/450;
    }
    else if (_flightMode == SATELLITE){
	rotAngle = (origAngle - (origAngle/100.0)*pow((t-10),2))/300;
    }

    if(angle > 0.0) {
    	osg::Vec3 crsVec = from^to;
    	crsVec.normalize();

    	osg::Matrix rotM;
    	rotM.makeRotate(DegreesToRadians(-1*rotAngle),crsVec);

  	angle -= rotAngle;
    	_rotMat = rotM * objMat;
    }

    else {
	_rotMat = objMat;
    }

    if(_flightMode == SATELLITE) {
    	zoomOut(trans1, _rotMat);
    }

    else if (_flightMode == AIRPLANE){
	rise(trans1, _rotMat);
    }

    else {
        trans1[1] = minHeight;
	_rotMat.setTrans(trans1);
	SceneManager::instance()->setObjectMatrix(_rotMat);
    }
    
}

CameraFlight::CameraFlight()
{
    std::cerr << "CameraFlight menu created" << std::endl;
}

CameraFlight::~CameraFlight()
{

}

bool CameraFlight::init()
{
    std::cerr << "CameraFlight init()" << std::endl;

    /*** Menu Setup ***/
    _camMenu = new cvr::SubMenu("Camera Flight", "Camera Flight");
    _camMenu->setCallback(this);
    cvr::PluginHelper::addRootMenuItem(_camMenu);

    _algoMenu = new cvr::SubMenu("Algorithm", "Algorithm");
    _algoMenu->setCallback(this);
    _camMenu->addItem(_algoMenu);

    _destMenu = new cvr::SubMenu("Destination", "Destination");
    _destMenu->setCallback(this);
    _camMenu->addItem(_destMenu);

    _instant = new cvr::MenuCheckbox("Instant", true);
    _instant->setCallback(this);
    _algoMenu->addItem(_instant);

    _satellite = new cvr::MenuCheckbox("Satellite", false);
    _satellite->setCallback(this);
    _algoMenu->addItem(_satellite);

    _airplane = new cvr::MenuCheckbox("Airplane", false);
    _airplane->setCallback(this);
    _algoMenu->addItem(_airplane);
	
    _dest1 = new cvr::MenuCheckbox("UCSD", false);
    _dest1->setCallback(this);
    _destMenu->addItem(_dest1);

    _dest2 = new cvr::MenuCheckbox("Tokyo", false);
    _dest2->setCallback(this);
    _destMenu->addItem(_dest2);

    _dest3 = new cvr::MenuCheckbox("New York", false);
    _dest3->setCallback(this);
    _destMenu->addItem(_dest3);

    _dest4 = new cvr::MenuCheckbox("Sydney Harbor", false);
    _dest4->setCallback(this);
    _destMenu->addItem(_dest4);

    _dest5 = new cvr::MenuCheckbox("Rome", false);
    _dest5->setCallback(this);
    _destMenu->addItem(_dest5);

    _dest6 = new cvr::MenuCheckbox("Fullerton", false);
    _dest6->setCallback(this);
    _destMenu->addItem(_dest6);

    _customDestMenu = new cvr::SubMenu("Custom","Custom");
    _customDestMenu->setCallback(this);
    _destMenu->addItem(_customDestMenu);

    _customLat = new cvr::MenuRangeValue("Latitude", -90.0, 90.0, 0.0, 0.01);
    _customLat->setCallback(this);
    _customDestMenu->addItem(_customLat);

    _customLon = new cvr::MenuRangeValue("Longitude", -180.0, 180.0, 0.0, 0.01);
    _customLon->setCallback(this);
    _customDestMenu->addItem(_customLon);

    _goButton = new cvr::MenuButton("Navigate");
    _goButton->setCallback(this);
    _customDestMenu->addItem(_goButton);
    /*** Menu Setup Finish ***/
	
    activeMode = _instant;
    _flightMode = INSTANT;
    destMode = NULL;
    return true;
}

void CameraFlight::menuCallback(cvr::MenuItem * item)
{
    if (item == _instant)
    {
	if(_flightMode == AIRPLANE)
	    normalView();
	if(activeMode != item)
	{
	    activeMode->setValue(false);
	    std::cerr<<"Instant Transition has selected"<<std::endl;
	}

	activeMode = _instant;
	_instant->setValue(true);
	_flightMode = INSTANT;
    }

    else if (item == _satellite)
    {
	if(_flightMode == AIRPLANE)
	    normalView();
	if(activeMode != item)
	{
	    activeMode->setValue(false);
	    std::cerr<<"Satellite Transition has selected"<<std::endl;
	}

	activeMode = _satellite;
	_satellite->setValue(true);
	_flightMode = SATELLITE;
    }

    else if (item == _airplane)
    {
	if(_flightMode != AIRPLANE){
	    planeView();
	}
	if(activeMode != item)
	{
	    activeMode->setValue(false);
	    std::cerr<<"Airplane Transition has selected"<<std::endl;
	}

	activeMode = _airplane;
	_airplane->setValue(true);
	_flightMode = AIRPLANE;
    }

    else if (item == _dest1)
    {
	if(destMode != item && destMode != NULL)
	{
	    destMode->setValue(false);
	}

	destMode = _dest1;
	_dest1->setValue(true);
	_destVec.set(0.573827, -2.04617, 0.0);
	navigate( _destVec);

	crsV1 = PluginHelper::getWorldToObjectTransform().getTrans();
	crsV1.normalize();
	
    }

    else if (item == _dest2)
    {
	if(destMode != item && destMode != NULL)
	{
	    destMode->setValue(false);
	}

	destMode = _dest2;
	_dest2->setValue(true);
	_destVec.set(0.622566, 2.43884, 0.0);
	navigate( _destVec);

	crsV1 = PluginHelper::getWorldToObjectTransform().getTrans();
	crsV1.normalize();
    }

    else if (item == _dest3)
    {
	if(destMode != item && destMode != NULL)
	{
	    destMode->setValue(false);
	}

	destMode = _dest3;
	_dest3->setValue(true);
	_destVec.set(0.710774,-1.29189, 0.0);
	navigate( _destVec);
    }

    else if (item == _dest4)
    {
	if(destMode != item && destMode != NULL)
	{
	    destMode->setValue(false);
	}

	destMode = _dest4;
	_dest4->setValue(true);
	_destVec.set(-0.590719, 2.63979, 0.0);
	navigate( _destVec);
    }

    else if (item == _dest5)
    {
	if(destMode != item && destMode != NULL)
	{
	    destMode->setValue(false);
	}

	destMode = _dest5;
	_dest5->setValue(true);
	_destVec.set(0.67315, 0.389608, 0.0);
	navigate( _destVec);
    }
 
    else if (item == _dest6)
    {
	if(destMode != item && destMode != NULL)
	{
	    destMode->setValue(false);
	}

	destMode = _dest6;
	_dest6->setValue(true);
	_destVec.set(0.590992, -2.05847, 0.0);
	navigate( _destVec);
    }

    else if(item == _goButton)
    {
	float lat, lon;
	if(destMode != NULL)
	{
	    destMode->setValue(false);
	}

	destMode = NULL;
	lat = _customLat->getValue();	
	lon = _customLon->getValue();
	_destVec.set(DegreesToRadians(lat),DegreesToRadians(lon), 0.0);
	navigate( _destVec);
    }
}


void CameraFlight::preFrame()
{
    w2o = SceneManager::instance()->getWorldToObjectTransform();
    o2w = SceneManager::instance()->getObjectToWorldTransform();

    osgEarth::MapNode* mapNode = MapNode::findMapNode(SceneManager::instance()->getObjectsRoot());
    map = mapNode->getMap();

    origPlanetPoint = PluginHelper::getWorldToObjectTransform().getTrans();

    if(ComController::instance()->isMaster())
    {
	map->getProfile()->getSRS()->getEllipsoid()->convertXYZToLatLongHeight(
					origPlanetPoint.x(),
					origPlanetPoint.y(),
					origPlanetPoint.z(),
					latLonHeight.x(),
					latLonHeight.y(),
					latLonHeight.z());
	latLonHeight[2] = 0.0;

	map->getProfile()->getSRS()->getEllipsoid()->convertLatLongHeightToXYZ(
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

    if(flagRot){
	t += cvr::PluginHelper::getLastFrameDuration();
	if(_flightMode == INSTANT) {
	    rotate(fromVec,toVec);
	}

	if(_flightMode == SATELLITE) {
	    if(t >= 30)
		if(angle > 0.001) {
		    rotate(fromVec,toVec);
		}
		else
		    flagRot = false;
	    else
		rotate(fromVec, toVec);
	}

	else if(_flightMode == AIRPLANE) {
	    if(angle <= 0.0)
		flagRot = false;
	    else
	    	rotate(fromVec, toVec);
	}
    }

    if(direction) {
	directSet(rotdir);
    }
}

void CameraFlight::postFrame()
{

}

bool CameraFlight::processEvent(InteractionEvent * event)
{
    TrackedButtonInteractionEvent * tie = event->asTrackedButtonEvent();
    KeyboardInteractionEvent * kie = event->asKeyboardEvent();

    if(event->asMouseEvent())
    {
	
    }
    if(kie) 
    {
	if(kie->getInteraction() == KEY_DOWN)
	{
	    buttonEvent(kie->getKey());
	}
    }

    return false;
}

bool CameraFlight::processMouseEvent(MouseInteractionEvent * event)
{
    return false;
}

void CameraFlight::navigate(osg::Vec3 destVec)
{
    flagRot = false;
    direction = false;
    osg::Matrix objMat = SceneManager::instance()->getObjectTransform()->getMatrix();
    objMat.decompose(trans, rot, scale, so);

    map->getProfile()->getSRS()->getEllipsoid()->convertLatLongHeightToXYZ(
	destVec.x(),destVec.y(),destVec.z(),toVec.x(),toVec.y(),toVec.z());

    fromVec = origPlanetPoint;

    fromVec.normalize();
    toVec.normalize();

    origAngle = acos((fromVec * toVec)/((fromVec.length() * toVec.length())));	
    origAngle = RadiansToDegrees(origAngle);

    angle = origAngle;

  //  cout<<origAngle<<endl;
    switch(_flightMode)
    {
	case INSTANT:

	    flagRot = true;

	    break;
	
	case SATELLITE:

	    t = 0.0;

    	    if(origAngle <= 10)
		maxHeight = minHeight;
	    else 
		maxHeight = 1.5e+10;

	    flagRot = true;

	    break;

	case AIRPLANE:

	    t = 0.0;

	    maxHeight = findMaxHeight(origAngle);
	    curHeight = trans[2]; 
	    planeDir(fromVec, toVec);
	
	    break;

	default:
	    cout<<"PICK THE ALGORYTHM!!!!"<<endl;
	    break;
    }
}

bool CameraFlight::buttonEvent(int type)
{
    osg::Matrix curMatrix = SceneManager::instance()->getObjectTransform()->getMatrix();
    curMatrix.decompose(trans, rot, scale, so);
    double curScale = SceneManager::instance()->getObjectScale();

    if(type == 'p') {
	std::cerr<<"curMatrix"<<endl;
	printMat(curMatrix, curScale);

	cout<<"Trans = ";
	printVec(trans);

	cout<<"Scale = ";
	printVec(scale);

	cout<<"Rotate = ";
	printQuat(rot);

	cout<<"Scale Orient =";
	printQuat(so);

	std::cerr<<"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"<<endl;

	cout<<"x = "<<latLonHeight.x()<<", y = "<<latLonHeight.y()<<", z = "<<latLonHeight.z()<<endl;

	cout<<"x1 = "<<origPlanetPoint.x()<<", y1 = "<<origPlanetPoint.y()<<", z1 = "<<origPlanetPoint.z()<<endl;

	cout<<distanceToSurface<<endl;
	cout<<(trans[2] + distanceToSurface)<<endl;
	cout<<endl;
    }

    else if(type == 'a') {
	curMatrix.decompose(trans, rot, scale, so);
	trans[2] = trans[2] - 100000;
	curMatrix.setTrans(trans);
    	SceneManager::instance()->setObjectMatrix(curMatrix);	
    }

    else if(type == 'z') {
	curMatrix.decompose(trans, rot, scale, so);
	trans[2] = trans[2] + 100000;
	curMatrix.setTrans(trans);
    	SceneManager::instance()->setObjectMatrix(curMatrix);	
    }

    else if(type == 's') {
	_origMatrix = SceneManager::instance()->getObjectTransform()->getMatrix();
	_origScale = SceneManager::instance()->getObjectScale();
    }

    else if(type == 'r') {

    	objMat = SceneManager::instance()->getObjectTransform()->getMatrix();
    	objMat.decompose(trans1, rot1, scale1, so1);

	osg::Vec3 planetPoint;

	fromVec = origPlanetPoint;
	rotAngle = 10;

    	osg::Matrix rotM;
    	rotM.makeRotate(DegreesToRadians((-1)*rotAngle),fromVec);

    	_rotMat = rotM * objMat;
	
	SceneManager::instance()->setObjectMatrix(_rotMat);
    }

    else if(type == 't') {

    	objMat = SceneManager::instance()->getObjectTransform()->getMatrix();
    	objMat.decompose(trans1, rot1, scale1, so1);

	fromVec = origPlanetPoint;
	
	rotAngle = 10;

    	osg::Matrix rotM;
    	rotM.makeRotate(DegreesToRadians(1*rotAngle),fromVec);

    	_rotMat = rotM * objMat;

	SceneManager::instance()->setObjectMatrix(_rotMat);
    }

    else if(type == 'q') {
	cout<<"Distance = "<<distanceToSurface<<endl;
    }

    else if(type == 'g') {
	flagRot = false;
	flagZoom = true;
	osg::Matrix objMat = SceneManager::instance()->getObjectTransform()->getMatrix();
	
    	objMat.decompose(trans, rot, scale, so);
	double LFD, PD, FS, PS;

	a = (maxHeight - trans[1])/25.0;
	t = 0.0;
	total = 0.0;
    }

    else if(type == 'v') {
	objMat = SceneManager::instance()->getObjectTransform()->getMatrix();
    	objMat.decompose(trans1, rot1, scale1, so1);
	origPlanetPoint.normalize();

	osg::Matrix rotn;

	osg::Matrix rotM;

	osg::Vec3 v(origPlanetPoint.y(), origPlanetPoint.x(), 0);
	osg::Vec3 v1;
	v1.set(rotn.postMult(v));
	rotM.makeRotate(DegreesToRadians(-10.0),osg::Vec3(1,0,0));
	
	_rotMat = objMat * rotM;

	SceneManager::instance()->setObjectMatrix(_rotMat);
    }
    return false;
}


