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

CVRPLUGIN(CameraFlight)

using namespace std;
using namespace osg;
using namespace cvr;
using namespace osgEarth;
osg::Vec3 trans1, scale1, trans2, scale2;
osg::Vec3d latLonHeight, origPlanetPoint;
osg::Vec3d fromVec, toVec;
osg::Quat rot1, so1, rot2, so2;
time_t tstart, tend;

osg::Matrix _origMatrix;
osg::Matrix _zoomMat;
osg::Matrix _rotMat;

double _origScale;
double origAngle, rotAngle, angle;
double zIn = 0.0; 
double zOut = 0.0;
double zoomSpd = 0.0;
double distanceToSurface = 0.0;
double curScale = 0.0;
double maxHeight = 2.0e+10;
double minHeight = 6.5e+9;
double a = 0.0;
double b = 0.0;
double t = 0.0;
double total = 0.0;

osg::Matrix _destMat;
osg::Vec3 _destVec;
osg::Matrix objMat;
osg::Matrix _destMat1(0.911223, 0.385116, 0.146142, 0.0,
		      -0.400983, 0.748153, 0.528658, 0.0,
		      0.0942582, -0.540326, 0.83626, 0.0,
		      -1.1348e-06, 6.37481e+09, 0.0, 1.0);
osg::Matrix _destMat2(-0.603846, 0.621304, 0.499351, 0.0,
			-0.793729, -0.526245, -0.305054, 0.0,
			0.073246, -0.580558, 0.810918, 0.0,
			-2.04593e-05, 6.41844e+09, 0.0, 1.0);
osg::Matrix _destMat3(0.904779, -0.00145363, 0.425878, 0.0, 
			0.425043, -0.0596121, -0.903208, 0.0, 
			0.0267004, 0.998221, -0.0533179, 0.0, 
			0.0, 1.91344e+10, 0.0, 1.0);
osg::Matrix _destMat4(-0.673689, 0.729656, 0.117241, 0.0, 
			-0.281838, -0.400326, 0.871956, 0.0, 
			0.683163, 0.554384, 0.475339, 0.0, 
			0.0, 6.38457e+09, 0.0, 1.0);
osg::Matrix _destMat5(-0.183746, -0.729452, -0.658891, 0.0, 
			0.982294, -0.161188, -0.0954832, 0.0, 
			-0.0365551, -0.664769, 0.746154, 0.0, 
			0.0, 6.38774e+09, 0.0, 1.0);
osg::Matrix _destMat6(0.909811, 0.389903, 0.142195, 0.0, 
			-0.400523, 0.73511, 0.546987, 0.0, 
			0.108743, -0.554607, 0.824977, 0.0, 
			7.54296e-06, 6.37619e+09, 0.0, 1.0);
osg::Matrix o2w, w2o, curMatrix, rotMat;
bool check = false;
bool flagOut = false;
bool flagIn = false;
bool flagZoom = false;
bool flagRot = false;

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

void CameraFlight::zoomOut(osg::Vec3 v, osg::Matrix mat)
{
    if(t < 5) {
	v[1] = (-a*pow((t-5),2)) + maxHeight;

    }

    else {
	a = (maxHeight - minHeight)/25.0;
	v[1] = (-a*pow((t-5),2)) + maxHeight;
    }

    mat.setTrans(v);
    SceneManager::instance()->setObjectMatrix(mat);

}

void CameraFlight::rotate(osg::Vec3 from, osg::Vec3 to)
{
    objMat = SceneManager::instance()->getObjectTransform()->getMatrix();
    objMat.decompose(trans1, rot1, scale1, so1);

    rotAngle = (origAngle + (-(origAngle/25)*pow((t-5),2)))/195;

    if(angle > 0.0) {

    	osg::Vec3 crsVec = from^to;
    	crsVec.normalize();

    	osg::Matrix rotM;
    	rotM.makeRotate(DegreesToRadians(-1*rotAngle),crsVec);

  	angle -= rotAngle;
    	_rotMat = rotM * objMat;
	//SceneManager::instance()->setObjectMatrix(_rotMat);
    }

    else {
	_rotMat = objMat;
    }

    zoomOut(trans1,_rotMat);
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
	
    _reset = new cvr::MenuCheckbox("Reset", false);
    _reset->setCallback(this);
    _algoMenu->addItem(_reset);

    _dest1 = new cvr::MenuCheckbox("UCSD", false);
    _dest1->setCallback(this);
    _destMenu->addItem(_dest1);

    _dest2 = new cvr::MenuCheckbox("Tokyo", false);
    _dest2->setCallback(this);
    _destMenu->addItem(_dest2);

    _dest3 = new cvr::MenuCheckbox("South Pole", false);
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
	if(activeMode != item)
	{
	    activeMode->setValue(false);
	    std::cerr<<"Satellite Transition has selected"<<std::endl;
	}

	activeMode = _satellite;
	_satellite->setValue(true);
	_flightMode = SATELLITE;
    }

    else if (item == _reset)
    {
	if(activeMode != item)
	{
	    activeMode->setValue(false);
	    std::cerr<<"Reset Back to original"<<std::endl;
	}

	activeMode = _reset;
	_reset->setValue(true);
	
	SceneManager::instance()->setObjectMatrix(_origMatrix);
	SceneManager::instance()->setObjectScale(_origScale);
    }

    else if (item == _dest1)
    {
	if(destMode != item && destMode != NULL)
	{
	    destMode->setValue(false);
	    _dest1->setValue(true);
	}

	if(destMode != item) {
	    destMode = _dest1;
	    _destMat.set(_destMat1);
	    _destVec.set(0.573827, -2.04617, 0.0);
	    navigate(_destMat, _destVec);
	}
	else {
	    destMode = _dest1;
	}
	
    }

    else if (item == _dest2)
    {
	if(destMode != item && destMode != NULL)
	{
	    destMode->setValue(false);
	}

	destMode = _dest2;
	_dest2->setValue(true);
	_destMat.set(_destMat2);
	_destVec.set(0.622566, 2.43884, 0.0);
	navigate(_destMat, _destVec);
    }

    else if (item == _dest3)
    {
	if(destMode != item && destMode != NULL)
	{
	    destMode->setValue(false);
	}

	destMode = _dest3;
	_dest3->setValue(true);
	_destMat.set(_destMat3);
	_destVec.set(-1.51126, 1.54642, 0.0);
	navigate(_destMat, _destVec);
    }

    else if (item == _dest4)
    {
	if(destMode != item && destMode != NULL)
	{
	    destMode->setValue(false);
	}

	destMode = _dest4;
	_dest4->setValue(true);
	_destMat.set(_destMat4);
	_destVec.set(-0.590719, 2.63979, 0.0);
	navigate(_destMat, _destVec);
    }

    else if (item == _dest5)
    {
	if(destMode != item && destMode != NULL)
	{
	    destMode->setValue(false);
	}

	destMode = _dest5;
	_dest5->setValue(true);
	_destMat.set(_destMat5);
	_destVec.set(0.67315, 0.389608, 0.0);
	navigate(_destMat, _destVec);
    }
 
    else if (item == _dest6)
    {
	if(destMode != item && destMode != NULL)
	{
	    destMode->setValue(false);
	}

	destMode = _dest6;
	_dest6->setValue(true);
	_destMat.set(_destMat6);
	_destVec.set(0.590992, -2.05847, 0.0);
	navigate(_destMat, _destVec);
    }
}


void CameraFlight::preFrame()
{
    w2o = SceneManager::instance()->getWorldToObjectTransform();
    o2w = SceneManager::instance()->getObjectToWorldTransform();

    curMatrix = SceneManager::instance()->getObjectTransform()->getMatrix();
    curScale = SceneManager::instance()->getObjectScale();

    _zoomMat = SceneManager::instance()->getObjectTransform()->getMatrix();
    _zoomMat.decompose(trans1, rot1, scale1, so1);

    osgEarth::MapNode* mapNode = MapNode::findMapNode(SceneManager::instance()->getObjectsRoot());
    map = mapNode->getMap();

    t+=cvr::PluginHelper::getLastFrameDuration();

    if(ComController::instance()->isMaster())
    {
	origPlanetPoint = PluginHelper::getWorldToObjectTransform().getTrans();

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
	if(t >= 10)
	    flagRot = false;
	else
	    rotate(fromVec, toVec);
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

void CameraFlight::navigate(osg::Matrix destMat, osg::Vec3 destVec)
{
    osg::Matrix objMat = SceneManager::instance()->getObjectTransform()->getMatrix();

    switch(_flightMode)
    {
	case INSTANT:{
	    cout<<"USING INSTANT"<<endl;
	    SceneManager::instance()->setObjectMatrix(destMat);
	    break;
	}
	case SATELLITE:
	    cout<<"USING SATELLITE"<<endl;

	    t = 0.0;
	    total = 0.0;
	
    	    objMat.decompose(trans2, rot2, scale2, so2);
	    a = (maxHeight - trans2[1])/25.0;

	    map->getProfile()->getSRS()->getEllipsoid()->convertLatLongHeightToXYZ(
					destVec.x(),destVec.y(),destVec.z(),toVec.x(),toVec.y(),toVec.z());

	    fromVec = origPlanetPoint;

	    fromVec.normalize();
	    toVec.normalize();

	    origAngle = acos((fromVec * toVec)/((fromVec.length() * toVec.length())));	
	    origAngle = RadiansToDegrees(origAngle);

	    angle = origAngle;

    	    if(origAngle <= 10) {
		maxHeight = 6.5e+9;
	    }

	    else {
		maxHeight = 2.0e+10;
	    }

	    flagRot = true;
	    break;
	case AIRPLANE:
	    cout<<"USING AIRPLANE"<<endl;
	    break;
	default:
	    cout<<"PICK THE ALGORYTHM!!!!"<<endl;
	    break;
    }
}

bool CameraFlight::buttonEvent(int type/*, const osg::Matrix & mat*/)
{
//    osg::Matrix curMatrix = SceneManager::instance()->getObjectTransform()->getMatrix();
//    double curScale = SceneManager::instance()->getObjectScale();

 //   osg::Matrix w2o = SceneManager::instance()->getWorldToObjectTransform();
 //   osg::Matrix o2w = SceneManager::instance()->getObjectToWorldTransform();

    if(type == 'p') {
	std::cerr<<"curMatrix"<<endl;
	printMat(curMatrix, curScale);
	
	cout<<"x = "<<latLonHeight.x()<<", y = "<<latLonHeight.y()<<", z = "<<latLonHeight.z()<<endl;
//	std::cerr<<"WorldToObject"<<endl;
//	printMat(w2o, curScale);

//	std::cerr<<"ObjectToWorld"<<endl;
//	printMat(o2w, curScale);
    }

    else if(type == 'd') {
	curMatrix.decompose(trans1, rot1, scale1, so1);

	std::cerr<<"<<<<<<<<<<<<<<<<<<<<<<<<<<<<"<<endl;
	cout<<"Trans = ";
	printVec(trans1);

	cout<<"Scale = ";
	printVec(scale1);

	cout<<"Rotate = ";
	printQuat(rot1);

	cout<<"Scale Orient =";
	printQuat(so1);

	std::cerr<<"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"<<endl;
	    //osg::Matrix _trans = osg::Matrix::rotate(oldPos,currentPos) * osg::Matrix::translate(trans);
	
	    
	    //osg::Matrix rotMat = osg::Matrix::rotate(oldPos,currentPos);
	    //osg::Matrix wr = o2w * rotMat* w2o; 
//	    osg::Matrix _tmp = w2o * osg::Matrix::rotate(oldPos,currentPos) * o2w;
	    //osg::Matrix _scale = osg::Matrix::scale(scale);
	    //osg::Matrix _temp = osg::Matrix::translate(-trans) * _scale * _trans;

    }

    else if(type == 't') {
	curMatrix.setTrans(osg::Vec3(0.0,1e+09/*6.41844e+09*/,0));
	SceneManager::instance()->setObjectMatrix(curMatrix);
    }

    else if(type == 's') {
	_origMatrix = SceneManager::instance()->getObjectTransform()->getMatrix();
	_origScale = SceneManager::instance()->getObjectScale();
    }

    else if(type == 'z') {
	tstart = time(0);
//	zIn = 1e+10; 
//	zOut = 1e+10;
	cout<<"zpressed"<<endl;
	if (flagZoom == false)
	    flagZoom = true;
    }

    else if(type == 'r') {

	if (flagRot == false) {
	    flagRot = true;
	}
	else {
	    flagRot = false;
	}

	//cout<<"Old Matrix"<<endl;
	//curMatrix = SceneManager::instance()->getObjectTransform()->getMatrix();
	//curScale = SceneManager::instance()->getObjectScale();

//	printMat(curMatrix, curScale);

	
	/*osg::Matrix mat2 = */

        osg::Matrix rotM;	
	rotM.makeRotate(DegreesToRadians(1.0),osg::Vec3(0,1,0)); 
//	printMat(rotM, curScale);
	
	curMatrix= o2w * rotM * w2o;
	
  //      printMat(curMatrix, curScale);

	//curMatrix.setTrans(trans);
	//cout<<"New Matrix"<<endl;
	//printMat(curMatrix, curScale);
	cout<<"x = "<<origPlanetPoint[0]<<", y = "<<origPlanetPoint[1]<<", z = "<<origPlanetPoint[2]<<endl;

	osg::Matrix objMat = SceneManager::instance()->getObjectTransform()->getMatrix();

	objMat.decompose(trans1,rot1,scale1,so1);
	_destMat1.decompose(trans2,rot2,scale2,so2);
//	cout<<"rotate from"<<endl;
//	printQuat(rot1);
//	cout<<"rotate to"<<endl;
//	printQuat(rot2);
	osg::Vec3 vect1, vect2;
	double ang1, ang2;
	rot1.getRotate(ang1, vect1);
	rot2.getRotate(ang2, vect2);
	osg::Vec3 vec1 = rot1.asVec3();
	osg::Vec3 vec2 = rot2.asVec3();
	
	printVec(vect1);
	printVec(vect2);
	osg::Quat rotQuat;
	rotQuat.makeRotate(vect1, vect2);
//	printQuat(rotQuat);

	rotM.makeRotate(rotQuat); 
	osg::Matrix mat = objMat* rotM;
	mat.setTrans(trans1);
	SceneManager::instance()->setObjectMatrix(mat);
    }

    else if(type == 'q') {
	cout<<"Distance = "<<distanceToSurface<<endl;
    }

    else if(type == 'g') {
	flagOut = false;
	flagIn = false;
	flagRot = false;
	flagZoom = true;
	osg::Matrix objMat = SceneManager::instance()->getObjectTransform()->getMatrix();
	
/*      osg::Matrix rotM8;
	rotM8.makeRotate(DegreesToRadians(10.0),osg::Vec3(0,1,0));
	osg::Matrix mat9 = rotM8 * objMat;

	SceneManager::instance()->setObjectMatrix(mat9);
*/

    	objMat.decompose(trans2, rot2, scale2, so2);
	double LFD, PD, FS, PS;

	a = (maxHeight - trans2[1])/25.0;
	t = 0.0;
	total = 0.0;
    }

    else if(type == 'a') {
	SceneManager::instance()->setObjectMatrix(_origMatrix);
    }

    else if(type == 'm') {

 	if(flagRot) {
	    flagRot = false;
	}

	else {
	tstart = time(0);

	zIn = 1e+10; 
	zOut = 1e+10;

	osg::Matrix objMat = SceneManager::instance()->getObjectTransform()->getMatrix();

	
	osg::Vec3d tolatLon(0.573827, -2.04617,0);
	osg::Vec3d tolatLon1(0.622566, 2.43884, 0);
	osg::Vec3d toVec1, toVec2;

	map->getProfile()->getSRS()->getEllipsoid()->convertLatLongHeightToXYZ(
					tolatLon.x(),tolatLon.y(),tolatLon.z(),
					toVec1.x(),toVec1.y(),toVec1.z());

//	map->getProfile()->getSRS()->getEllipsoid()->convertLatLongHeightToXYZ(
//					tolatLon1.x(),tolatLon1.y(),tolatLon1.z(),
//					toVec2.x(),toVec2.y(),toVec2.z());

	fromVec = origPlanetPoint;

	toVec1.normalize();
//	toVec2.normalize();
	fromVec.normalize();
/*
	osg::Vec3 offset(0.0,1.0,0.0);

	offset = offset - fromVec;
	fromVec = fromVec + offset;
	toVec1 = toVec1 + offset;
	toVec2 = toVec2 + offset;


	printVec(fromVec);
        printVec(toVec1);
        printVec(toVec2);
*/
	cout<<endl;
	toVec = toVec1;

	double dot = fromVec * toVec;
	angle = acos(dot/((fromVec.length() * toVec.length())));	

	angle = RadiansToDegrees(angle);

	rotAngle = angle/100.0;
	cout<<angle<<endl; 	
	flagRot = true;
}
//	osg::Vec3 crsVec = toVec1^toVec2;
//	crsVec.normalize();

//	cout<<"Where you at"<<endl;
//	printVec(fromVec);

//	cout<<"UCSD"<<endl;
//	printVec(toVec1);

//	cout<<"Tokyo"<<endl;
//	printVec(toVec2);

	//osg::Vec3 rotVec = (fromVec ^ toVec);
	//rotVec.normalize();
	//origPlanetPoint.normalize();
	//latLonHeight.normalize();

//	printVec(crsVec);	
 //       osg::Matrix rotM;
//	rotM.makeRotate(DegreesToRadians(1.0),crsVec);



	//printVec(origPlanetPoint);
//	printVec(origPlanetPoint);
//	printVec(latLonHeight);
//	printMat(rotM, curScale);
	//objMat.decompose(trans1,rot1,scale1,so1);
	//osg::Vec3 vect1, vect2;
	//double ang1, ang2;
	//rot1.getRotate(ang1, vect1);
	//printVec(vect1);
	
//	osg::Matrix mat = objMat* rotM;
//	mat.setTrans(trans1);
//	SceneManager::instance()->setObjectMatrix(mat);
    }

    else if(type == 'x') {
	cout<<"DRAWlING"<<endl;
	osgEarth::MapNode* outMapNode = MapNode::findMapNode(SceneManager::instance()->getObjectsRoot());

	outputMap = outMapNode->getMap();

	double lat = 0.0;
	double lon = -1.5708;
	double hght = 0.0;

	osg::Matrix objMat = SceneManager::instance()->getObjectTransform()->getMatrix();

	
	osg::Vec3d tolatLon(0.573827, -2.04617,0);
	osg::Vec3d tolatLon1(0.622566, 2.43884, 0);
	osg::Vec3d toVec1, toVec2;

	map->getProfile()->getSRS()->getEllipsoid()->convertLatLongHeightToXYZ(
					lat,lon,hght,
					toVec1.x(),toVec1.y(),toVec1.z());

	osg::Matrixd output, output2;
	map->getProfile()->getSRS()->getEllipsoid()->computeLocalToWorldTransformFromXYZ(
		toVec1.x(), toVec1.y(), toVec1.z(), output);
	
	map->getProfile()->getSRS()->getEllipsoid()->computeLocalToWorldTransformFromXYZ(
		lat, lon, hght, output2);
	printMat(output,10.0);
	printMat(output2,10.0);

	osg::Geode * geode = new osg::Geode();
	osg::Vec3 centerVec(0.0,0.0,-20000.0);
	osg::ShapeDrawable* shape = new osg::ShapeDrawable(
		new osg::Cylinder(centerVec,10000.0, 2000000.0));
	geode->addDrawable(shape);
	
	osg::MatrixTransform * mat1 = new osg::MatrixTransform();

	mat1->setMatrix(output);
	mat1->addChild(geode);
	
	SceneManager::instance()->getObjectsRoot()->addChild(mat1);

	map->getProfile()->getSRS()->getEllipsoid()->convertLatLongHeightToXYZ(
					tolatLon.x(),tolatLon.y(),tolatLon.z(),
					toVec2.x(),toVec2.y(),toVec2.z());

	osg::Matrixd output1;
//	lat = 0.573827;
//	lon = -2.04617;
	map->getProfile()->getSRS()->getEllipsoid()->computeLocalToWorldTransformFromXYZ(
		toVec2.x(), toVec2.y(), toVec2.z(), output1);

	osg::Geode * geode1 = new osg::Geode();
	osg::Vec3 centerVec1(0.0,0.0,-20000.0);
	osg::ShapeDrawable* shape1 = new osg::ShapeDrawable(
		new osg::Cylinder(centerVec1,10000.0, 2000000.0));
	geode1->addDrawable(shape1);
	
	osg::MatrixTransform * mat2 = new osg::MatrixTransform();

	mat2->setMatrix(output1);
	mat2->addChild(geode1);
	
	SceneManager::instance()->getObjectsRoot()->addChild(mat2);
	//geode->addDrawable(shape1);
    }

    return false;

}

