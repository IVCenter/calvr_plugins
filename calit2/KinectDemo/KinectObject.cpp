#include "KinectObject.h"
#define VERTEXBIND 6

using namespace cvr;
using namespace std;
using namespace osg;

KinectObject::KinectObject(std::string name, std::string cloud_server, std::string skeleton_server, std::string color_server, std::string depth_server, osg::Vec3 position) : SceneObject(name, false, false, false, true, false)
{
    //Setup Extended SceneObject
    _kinectName = name;
    max_users = 12;
    useHands = false;
    _kinectFOV_on = false;
    switchNode = new osg::Switch();
    addChild(switchNode);
    _navigatable = true;//false;
    setNavigationOn(_navigatable);
    setMovable(false);
    //   kinectX = ConfigManager::getFloat("x", "Plugin.KinectDemo.KinectSkeleton", 0.0f);
    //   kinectY = ConfigManager::getFloat("y", "Plugin.KinectDemo.KinectSkeleton", 0.0f);
    //   kinectZ = ConfigManager::getFloat("z", "Plugin.KinectDemo.KinectSkeleton", 0.0f);
    kinectX = position.x();
    kinectY = position.y();
    kinectZ = position.z();
    cloudServer = cloud_server;
    skeletonServer = skeleton_server;
    depthServer = depth_server;
    colorServer = color_server;
    setPosition(position);
    //Setup Cloud
    cloudInit();
    skeletonOn();
    if(ConfigManager::getBool("Plugin.KinectDemo.ShowKinectFOV"))
    {
    _kinectFOV_on = true;
    showKinectFOV();
    }
    setupMenus();

    //Setup Skeleton
    //TODO

    //Add KinectModel
    if (ConfigManager::getBool("Plugin.KinectDemo.ShowKinectModel"))
    {
        //Loads Kinect Obj file
        Matrixd scale;
        double snum = 1;
        scale.makeScale(snum, snum, snum);
        MatrixTransform* modelScaleTrans = new MatrixTransform();
        modelScaleTrans->setMatrix(scale);
        _modelFileNode1 = osgDB::readNodeFile(ConfigManager::getEntry("Plugin.KinectDemo.3DModelFolder").append("kinect_mm.obj"));
        modelScaleTrans->addChild(_modelFileNode1);
        MatrixTransform* rotate = new osg::MatrixTransform();
        float rotDegrees[3];
        rotDegrees[0] = -90;
        rotDegrees[1] = 0;
        rotDegrees[2] = 180;
        rotDegrees[0] = DegreesToRadians(rotDegrees[0]);
        rotDegrees[1] = DegreesToRadians(rotDegrees[1]);
        rotDegrees[2] = DegreesToRadians(rotDegrees[2]);
        Quat rot = osg::Quat(rotDegrees[0], osg::Vec3d(1, 0, 0), rotDegrees[1], osg::Vec3d(0, 1, 0), rotDegrees[2], osg::Vec3d(0, 0, 1));
        Matrix rotMat;
        rotMat.makeRotate(rot);
        rotate->setMatrix(rotMat);
        rotate->addChild(modelScaleTrans);
        // MatrixTransform* translate = new osg::MatrixTransform();
        // osg::Matrixd tmat;
        // Vec3 pos = Vec3(kinectX, kinectY, kinectZ);
        // tmat.makeTranslate(pos);
        // translate->setMatrix(tmat);
        // translate->addChild(rotate);
        // switchNode->addChild(translate);
        switchNode->addChild(rotate);
    }

    cm = new CloudManager(cloudServer);
    _cameraOn = _depthOn = false;
}

void KinectObject::setupMenus()
{
    addMoveMenuItem();
    addNavigationMenuItem();
    loadTransformMenu();

    if(cvr::ConfigManager::getBool("Plugin.KinectDemo.TransformsAtStartup"))
    {
       for(int i=0; i < transform_path->size(); i++)
       {
         std::string name2 = transform_path->at(i);
         name2.erase(0,name2.length()-5);
         name2.erase(1,name2.length());
         if(name2 == _kinectName)
         {
           transformFromFile(transform_path->at(i));
           cerr << "Transformed: " << name2 << "\n";
	   break;
	 }
       }
    }

    SubMenu* usersMenu = new SubMenu("Users Menu");
    addMenuItem(usersMenu);
    for(int n=0; n < max_users; n++)
    {
     stringstream ss;
     ss << "User " <<  n;
     std::string userNum = ss.str();
     cvr::MenuCheckbox* toggleUsers = new MenuCheckbox(userNum, true);
    toggleUsers->setCallback(this);
    usersMenu->addItem(toggleUsers);
    _toggleUsersArray.push_back(toggleUsers);
    }

    _toggleKinectFOV = new MenuCheckbox("Kinect FOV",_kinectFOV_on);
    _toggleKinectFOV->setCallback(this);
    addMenuItem(_toggleKinectFOV);
}

void KinectObject::menuCallback(MenuItem* item)
{

    for (int i = 0; i < transform_list->size(); i++)
    {
        if (item == transform_list->at(i))
        {
            transformFromFile(transform_path->at(i));
            //cout << "found it " << transform_path->at(i) << endl;
            break;
        }
    }
    for(int n; n < _toggleUsersArray.size(); n++)
    {
    if (item == _toggleUsersArray[n])
    {
        if (_toggleUsersArray[n]->getValue())
        {
		 cm->userOn[n] = true;
	         _cloudGroups[n]->setNodeMask(0xffffffff);
        }
        else
        {
		 cm->userOn[n] = false;
	         _cloudGroups[n]->setNodeMask(0);
        }
    }
    }
    if (item == _toggleKinectFOV)
    {
        if (_toggleKinectFOV->getValue())
        {
		 _kinectFOV_on = true;
                 if(kinectFOV == NULL)
		 {
		   showKinectFOV();
		 }
		 else
		 {
                   switchNode->addChild(kinectFOV.get());
		 }
        }
        else
        {
		 _kinectFOV_on = false;
                 if(kinectFOV != NULL)
		 {
                   switchNode->removeChild(kinectFOV.get());
		 }
        }
    }
    SceneObject::menuCallback(item);
}


void KinectObject::transformFromFile(string filename)
{
    CalibrateKinect* calibrateTool = new CalibrateKinect();
    osg::Vec3Array* helmertVec3Array = calibrateTool->getTransformOutput(filename);

    if (helmertVec3Array->size() != 0)
    {
        Matrix calcMatrix;
        Vec3 r1 = helmertVec3Array->at(0);
        Vec3 r2 = helmertVec3Array->at(1);
        Vec3 r3 = helmertVec3Array->at(2);
        Vec3 r4 = Vec3(0, 0, 0);
        calcMatrix.set(r1.x(), r1.y(), r1.z(), 0, r2.x(), r2.y(), r2.z(), 0, r3.x(), r3.y(), r3.z(), 0, r4.x(), r4.y(), r4.z(), 1);
        Vec3 calcTranslate = helmertVec3Array->at(3);
        Vec3 kScale = helmertVec3Array->at(4);
        float scale = kScale.x();
        //Vec3 koPos = getPosition();
        Vec3 koPos = Vec3(0,0,0);
        calcTranslate = (calcTranslate + (calcMatrix * koPos * scale));
        Matrix inverseRot;
        calcMatrix = inverseRot.inverse(calcMatrix);
        setTransform(calcMatrix);
        setScale(kScale.x());
        setPosition(calcTranslate);
        helmertTArray.push_back(calcTranslate);
        helmertMArray.push_back(calcMatrix);
        helmertSArray.push_back(kScale.x());
        cerr << "File Present\n";
    }
}

void KinectObject::cloudInit()
{
    pgm1 = new osg::Program;
    pgm1->setName("Sphere");
    std::string shaderPath = ConfigManager::getEntry("Plugin.KinectDemo.ShaderPath");
    pgm1->addShader(osg::Shader::readShaderFile(osg::Shader::VERTEX, osgDB::findDataFile(shaderPath + "/Sphere.vert")));
    pgm1->addShader(osg::Shader::readShaderFile(osg::Shader::FRAGMENT, osgDB::findDataFile(shaderPath + "/Sphere.frag")));
    pgm1->addShader(osg::Shader::readShaderFile(osg::Shader::GEOMETRY, osgDB::findDataFile(shaderPath + "/Sphere.geom")));
    pgm1->addBindAttribLocation("morphvertex", VERTEXBIND);
    pgm1->setParameter(GL_GEOMETRY_VERTICES_OUT_EXT, 4);
    pgm1->setParameter(GL_GEOMETRY_INPUT_TYPE_EXT, GL_POINTS);
    pgm1->setParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP);
    initialPointScale = ConfigManager::getFloat("Plugin.KinectDemo.KinectDefaultOn.KinectPointSize");
    _firstRun = true;
    _cloudIsOn = false;
}
void KinectObject::cloudOn()
{
    if (_firstRun)
    {
        cout << "Starting Thread\n";
        //cm = new CloudManager(cloudServer);
        cout << "Started\n";
        cm->start();
        for(int i = 0; i < max_users; i++)
        {
        osg::Group* group = new osg::Group();
        osg::StateSet* state = group->getOrCreateStateSet();
        state->setAttribute(pgm1);
        state->addUniform(new osg::Uniform("pointScale", initialPointScale));
        state->addUniform(new osg::Uniform("globalAlpha", 1.0f));
        float pscale = initialPointScale;
        osg::Uniform*  _scaleUni = new osg::Uniform("pointScale", 1.0f * pscale);
        group->getOrCreateStateSet()->addUniform(_scaleUni);
        _cloudGroups.push_back(group);
        switchNode->addChild(group);
        }
        _cloudIsOn = true;
    }
    else
    {
        cerr << "Restarting\n";
        _firstRun = true;
        cm->should_quit = false;
        cm->start();
        _cloudIsOn = true;
    }
}

void KinectObject::cloudOff()
{
    //    if (cm != NULL)
    if (_cloudIsOn == true)
    {
        _cloudIsOn = false;
        cm->quit();
        for(int n; n < max_users; n++)
	{
        _cloudGroups[n]->removeChild(0,1);
	}
    }
}

void KinectObject::cloudUpdate()
{
    if (_cloudIsOn == false) return;

    if (true)
    {
        if (cm->firstRunStatus() != 0)
        {
            if (cm->kinectVertices != NULL)
            {
                if (cm->kinectVertices->size() != 0)
                {
                    if (_firstRun)
                    {
                        _firstRun = false;
                        for(int n=0; n < cm->max_users; n++)
			{
                        osg::Geode* kgeode = new osg::Geode();
                        kgeode->setCullingActive(false);
                        osg::Geometry* geom = new osg::Geometry();
                        StateSet* state = geom->getOrCreateStateSet();
                        state->setMode(GL_LIGHTING, StateAttribute::OFF | osg::StateAttribute::OVERRIDE);
                        osg::VertexBufferObject* vboP = geom->getOrCreateVertexBufferObject();
                        vboP->setUsage(GL_STREAM_DRAW);
                        //vboP->setUsage(GL_DYNAMIC_DRAW);
                        //vboP->setUsage(GL_STATIC_DRAW);
                        geom->setUseDisplayList(false);
                        geom->setUseVertexBufferObjects(true);

                        osg::DrawArrays* drawArray;
 			if(n > 0 && useHands)
                        {
                        drawArray = cm->drawArraysHand[n].get();
                        geom->addPrimitiveSet(drawArray);
                        //geom->setVertexArray(cm->kinectVertices.get());
                        geom->setVertexAttribArray(VERTEXBIND, cm->lHandVerticesArray[n].get());
                        geom->setVertexAttribBinding(VERTEXBIND, osg::Geometry::BIND_PER_VERTEX);
                        geom->setColorArray(cm->lHandColoursArray[n].get());
                        }
                        else
                        {
                        drawArray = cm->drawArrays[n].get();
                        geom->addPrimitiveSet(drawArray);
                        //geom->setVertexArray(cm->kinectVertices.get());
                        geom->setVertexAttribArray(VERTEXBIND, cm->userVerticesArray[n].get());
                        geom->setVertexAttribBinding(VERTEXBIND, osg::Geometry::BIND_PER_VERTEX);
                        geom->setColorArray(cm->userColoursArray[n].get());
                        }
                        geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
                        kgeode->addDrawable(geom);
                        _cloudGroups[n]->addChild(kgeode);
			}
                    }
                    else
                    {
            //geom->computeBound();
            //const osg::BoundingBox& bbox = geom->getBound();
            //userRadius[1] = bbox.radius();
            //osg::Vec3f center = bbox.center();
            //userRadius[1] = center.x();
           // cerr << "Center" << bbox.radius() << "\n";
                        //cm->kinectVertices.get()->dirty();
                        //cm->kinectColours.get()->dirty();
                        //osg::Array* temp = geom->getVertexArray();
                        //cerr << temp->getNumElements() << "\n";
                    }
                }
            }
        }
    }
}

void KinectObject::skeletonOn()
{
    context = new zmq::context_t(1);
    skel_frame = new RemoteKinect::SkeletonFrame();
    skel_socket = new SubSocket<RemoteKinect::SkeletonFrame>(*context, skeletonServer);
}

void KinectObject::skeletonUpdate()
{
    //    cerr << "." ;
    while (skel_socket->recv(*skel_frame))
    {
        //    cerr << "+";
        // remove all the skeletons that are no longer reported by the server
        for (std::map<int, Skeleton>::iterator it2 = mapIdSkel.begin(); it2 != mapIdSkel.end(); ++it2)
        {
            bool found = false;

            for (int i = 0; i < skel_frame->skeletons_size(); i++)
            {
                if (skel_frame->skeletons(i).skeleton_id() == it2->first)
                {
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                mapIdSkel[it2->first].detach(switchNode);
            }
        }

        if (skel_frame->skeletons_size() > 0)

            //  cerr << "Skels:" << skel_frame->skeletons_size() << "\n";
            // update all skeletons' joints' positions
            for (int i = 0; i < skel_frame->skeletons_size(); i++)
            {
                // Skeleton reported but not in the map -> create a new one
                if (mapIdSkel.count(skel_frame->skeletons(i).skeleton_id()) == 0)
                {
                    mapIdSkel[skel_frame->skeletons(i).skeleton_id()] = Skeleton(); ///XXX remove Skeleton(); part
                    // mapIdSkel[sf->skeletons(i).skeleton_id()].attach(_root);
                    // cerr << "Found Skeleton\n";
                    mapIdSkel[skel_frame->skeletons(i).skeleton_id()].attach(switchNode);
                }

                // Skeleton previously detached (stopped being reported), but is again reported -> reattach
                if (mapIdSkel[skel_frame->skeletons(i).skeleton_id()].attached == false)
                    mapIdSkel[skel_frame->skeletons(i).skeleton_id()].attach(switchNode);

                for (int j = 0; j < skel_frame->skeletons(i).joints_size(); j++)
                {
                    mapIdSkel[skel_frame->skeletons(i).skeleton_id()].update(
                        skel_frame->skeletons(i).joints(j).type(),
                        skel_frame->skeletons(i).joints(j).x(),
                        skel_frame->skeletons(i).joints(j).z(),
                        skel_frame->skeletons(i).joints(j).y(),
                        skel_frame->skeletons(i).joints(j).qx(),
                        skel_frame->skeletons(i).joints(j).qz(),
                        skel_frame->skeletons(i).joints(j).qy(),
                        skel_frame->skeletons(i).joints(j).qw());
                }
            }
    }
}

void KinectObject::skeletonOff()
{
}

std::map<int, Skeleton>* KinectObject::skeletonGetMap()
{
    return &mapIdSkel;
}

void KinectObject::cameraOn()
{
    bitmaptransform = new osg::MatrixTransform();
    osg::Vec3 pos(0, 0, 0);
    pos = Vec3(1000.0, -2000, 400);
    osg::Matrixd tmat;
    tmat.makeTranslate(pos);
    osg::Matrixd rmat;
    rmat.makeRotate(45, 1, 0, 0);
    osg::Matrix combined;
    combined.makeIdentity();
    combined.preMult(tmat);
    combined.preMult(rmat);
    bitmaptransform->setMatrix(combined);
    image = new osg::Image();
    pTex = new osg::Texture2D();
    pGeode = new osg::Geode();
    pStateSet = pGeode->getOrCreateStateSet();
    pStateSet->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    pStateSet->setTextureAttributeAndModes(0, pTex, osg::StateAttribute::ON);
    pGeode->setStateSet(pStateSet);
    geometry = new osg::Geometry();
    pGeode->addDrawable(geometry);
    vertexArray = new osg::Vec3Array();
    vertexArray->push_back(osg::Vec3(0, 0, 0));
    vertexArray->push_back(osg::Vec3(640 , 0, 0));
    vertexArray->push_back(osg::Vec3(640 , 480 , 0));
    vertexArray->push_back(osg::Vec3(0, 480 , 0));
    geometry->setVertexArray(vertexArray);
    colorArray = new osg::Vec4Array();
    colorArray->push_back(osg::Vec4(1.f, 1.f, 1.f, 1.f));
    geometry->setColorArray(colorArray);
    geometry->setColorBinding(osg::Geometry::BIND_OVERALL);
    texCoordArray = new osg::Vec2Array();
    texCoordArray->push_back(osg::Vec2(0.f, 0.f));
    texCoordArray->push_back(osg::Vec2(1.f, 0.f));
    texCoordArray->push_back(osg::Vec2(1.f, 1.f));
    texCoordArray->push_back(osg::Vec2(0.f, 1.f));
    geometry->setTexCoordArray(0, texCoordArray);
    geometry->addPrimitiveSet(new osg::DrawArrays(GL_TRIANGLE_FAN, 0, 4));
    bitmaptransform->addChild(pGeode);
    switchNode->addChild(bitmaptransform);
    color_socket = new SubSocket<RemoteKinect::ColorMap> (*context, colorServer);
    colm = new RemoteKinect::ColorMap();
    _cameraOn = true;
}

void KinectObject::cameraUpdate()
{
    if (_cameraOn == false) return;

    if (color_socket->recv(*colm))
    {
        for (int y = 0; y < 480; y++)
        {
            for (int x = 0; x < 640; x++)
            {
                uint32_t packed = colm->pixels(y * 640 + x);
                color_pixels[640 * (479 - y) + x] = packed;
            }
        }
    }

    image->setImage(640, 480, 1, GL_RGBA16, GL_RGBA, GL_UNSIGNED_BYTE, (unsigned char*) &color_pixels[0], osg::Image::NO_DELETE);
    pTex->setImage(image);
}

void KinectObject::cameraOff()
{
    if (_cameraOn == false) return;

    _cameraOn = false;
    switchNode->removeChild(bitmaptransform);

    if (color_socket) {
        delete color_socket;
        color_socket = NULL;
    }
}

void KinectObject::depthOn()
{
    minDistHSVDepth = 300;
    maxDistHSVDepth = 6000;
    depthBitmaptransform = new osg::MatrixTransform();
    osg::Vec3 pos(0, 0, 0);
    pos = Vec3(-2000.0, -2000, 400);
    osg::Matrixd tmat;
    tmat.makeTranslate(pos);
    osg::Matrixd rmat;
    rmat.makeRotate(45, 1, 0, 0);
    osg::Matrix combined;
    combined.makeIdentity();
    combined.preMult(tmat);
    combined.preMult(rmat);
    depthBitmaptransform->setMatrix(combined);
    depthImage = new osg::Image();
    depthPTex = new osg::Texture2D();
    depthPGeode = new osg::Geode();
    depthPStateSet = depthPGeode->getOrCreateStateSet();
    depthPStateSet->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    depthPStateSet->setTextureAttributeAndModes(0, depthPTex, osg::StateAttribute::ON);
    depthPGeode->setStateSet(depthPStateSet);
    depthGeometry = new osg::Geometry();
    depthPGeode->addDrawable(depthGeometry);
    depthVertexArray = new osg::Vec3Array();
    depthVertexArray->push_back(osg::Vec3(0, 0, 0));
    depthVertexArray->push_back(osg::Vec3(640 , 0, 0));
    depthVertexArray->push_back(osg::Vec3(640 , 480 , 0));
    depthVertexArray->push_back(osg::Vec3(0, 480 , 0));
    depthGeometry->setVertexArray(depthVertexArray);
    depthColorArray = new osg::Vec4Array();
    depthColorArray->push_back(osg::Vec4(1.f, 1.f, 1.f, 1.f));
    depthGeometry->setColorArray(depthColorArray);
    depthGeometry->setColorBinding(osg::Geometry::BIND_OVERALL);
    depthTexCoordArray = new osg::Vec2Array();
    depthTexCoordArray->push_back(osg::Vec2(0.f, 0.f));
    depthTexCoordArray->push_back(osg::Vec2(1.f, 0.f));
    depthTexCoordArray->push_back(osg::Vec2(1.f, 1.f));
    depthTexCoordArray->push_back(osg::Vec2(0.f, 1.f));
    depthGeometry->setTexCoordArray(0, depthTexCoordArray);
    depthGeometry->addPrimitiveSet(new osg::DrawArrays(GL_TRIANGLE_FAN, 0, 4));
    depthBitmaptransform->addChild(depthPGeode);
    switchNode->addChild(depthBitmaptransform);
    depth_socket = new SubSocket<RemoteKinect::DepthMap> (*context, depthServer);

    // precompute colors for ... mm distances
    for (int i = 0; i < 15001; i++) getColorRGBDepth(i);

    // precompute packed values for colors on depthmap
    for (int i = 0; i < 15000; i++)
    {
        //http://graphics.stanford.edu/~mdfisher/Kinect.html
        osg::Vec4 color = getColorRGBDepth(i);
        char rrr = (char)((float)color.r() * 255.0);
        char ggg = (char)((float)color.g() * 255.0);
        char bbb = (char)((float)color.b() * 255.0);
        uint32_t packed = (((rrr << 0) | (ggg << 8) | (bbb << 16)) & 0x00FFFFFF);
        dpmap[i] = packed;
    }

    depm = new RemoteKinect::DepthMap();
    _depthOn = true;
}

void KinectObject::depthUpdate()
{
    if (_depthOn == false) return;

    if (depth_socket->recv(*depm))
    {
        for (int y = 0; y < 480; y++)
        {
            for (int x = 0; x < 640; x++)
            {
                int val = depm->depths(y * 640 + x);
                //              if (dpmap.count(val) == 0)
                //              {
                //                  osg::Vec4 color = getColorRGBDepth(val);
                //                  char rrr = (char)((float)color.r() * 255.0);
                //                  char ggg = (char)((float)color.g() * 255.0);
                //                  char bbb = (char)((float)color.b() * 255.0);
                //                  uint32_t packed = (((rrr << 0) | (ggg << 8) | (bbb << 16)) & 0x00FFFFFF);
                //                  dpmap[val] = packed;
                //              }
                depth_pixels[640 * (479 - y) + x] = dpmap[val];//packed;
            }
        }
    }

    depthImage->setImage(640, 480, 1, GL_RGBA16, GL_RGBA, GL_UNSIGNED_BYTE, (unsigned char*) &depth_pixels[0], osg::Image::NO_DELETE);
    depthPTex->setImage(depthImage);
}

void KinectObject::depthOff()
{
    if (_depthOn == false) return;

    _depthOn = false;

    if (depth_socket) {
        delete depth_socket;
        depth_socket = NULL;
    }

    switchNode->removeChild(depthBitmaptransform);
}

osg::Vec4f KinectObject::getColorRGBDepth(int dist)
{
    if (distanceColorMapDepth.count(dist) == 0) // that can be commented out after precomputing completely if the range of Z is known (and it is set on the server side)
    {
        float r, g, b;
        float h = depth_to_hue(minDistHSVDepth, dist, maxDistHSVDepth);
        HSVtoRGB(&r, &g, &b, h, 1, 1);
        distanceColorMapDepth[dist] = osg::Vec4f(r, g, b, 1);
    }

    return distanceColorMapDepth[dist];
}
void KinectObject::toggleNavigation(bool navigatable)
{
    setNavigationOn(navigatable);
}
void KinectObject::showKinectFOV()
{

            if (_kinectFOV_on)
            {
              if(kinectFOV == NULL)
	      {
                //Draw Kinect FOV
                float width;
                float height;
                Vec3 offsetScreen = Vec3(0, 500, 0);
                Vec3 pos;
                Vec4f color = Vec4f(0, 0.42, 0.92, 1);
                //Create Quad Face
                width = 543;
                height = 394;
                pos = Vec3(-(width / 2), 0, -(height / 2));
                pos += Vec3(kinectX, kinectY, kinectZ);
                pos += offsetScreen;
                osg::Geometry* geo = new osg::Geometry();
                osg::Vec3Array* verts = new osg::Vec3Array();
                verts->push_back(pos);
                verts->push_back(pos + osg::Vec3(width, 0, 0));
                verts->push_back(pos + osg::Vec3(width, 0, height));
                verts->push_back(pos + osg::Vec3(0, 0, height));
                //do it Again
                width = 3800.6;
                height = 2756;
                offsetScreen = Vec3(0, 3500, 0);
                pos = Vec3(-(width / 2), 0, -(height / 2));
                pos += Vec3(kinectX, kinectY, kinectZ);
                pos += offsetScreen;
                verts->push_back(pos);
                verts->push_back(pos + osg::Vec3(width, 0, 0));
                verts->push_back(pos + osg::Vec3(width, 0, height));
                verts->push_back(pos + osg::Vec3(0, 0, height));
                //....................................
                int size = verts->size() / 2;
                Geometry* geom = new Geometry();
                Geometry* tgeom = new Geometry();
                Geode* fgeode = new Geode();
                Geode* lgeode = new Geode();
                geom->setVertexArray(verts);
                tgeom->setVertexArray(verts);

                for (int n = 0; n < size; n++)
                {
                    DrawElementsUInt* face = new DrawElementsUInt(PrimitiveSet::QUADS, 0);
                    face->push_back(n);
                    face->push_back(n + size);
                    face->push_back(((n + 1) % size) + size);
                    face->push_back((n + 1) % size);
                    geom->addPrimitiveSet(face);
                }

                StateSet* state(fgeode->getOrCreateStateSet());
                Material* mat(new Material);
                mat->setColorMode(Material::DIFFUSE);
                mat->setDiffuse(Material::FRONT_AND_BACK, color);
                state->setAttribute(mat);
                state->setRenderingHint(StateSet::OPAQUE_BIN);
                state->setMode(GL_BLEND, StateAttribute::ON);
                state->setMode(GL_LIGHTING, StateAttribute::OFF);
                osg::PolygonMode* polymode = new osg::PolygonMode;
                polymode->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::LINE);
                state->setAttributeAndModes(polymode, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
                fgeode->setStateSet(state);
                fgeode->addDrawable(geom);

                if (false)
                {
                    geo->setVertexArray(verts);
                    osg::DrawElementsUInt* ele = new osg::DrawElementsUInt(
                        osg::PrimitiveSet::QUADS, 0);
                    ele->push_back(0);
                    ele->push_back(1);
                    ele->push_back(2);
                    ele->push_back(3);
                    ele->push_back(4);
                    ele->push_back(5);
                    ele->push_back(6);
                    ele->push_back(7);
                    geo->addPrimitiveSet(ele);
                    Geode* fgeode = new Geode();
                    StateSet* state(fgeode->getOrCreateStateSet());
                    Material* mat(new Material);
                    mat->setColorMode(Material::DIFFUSE);
                    mat->setDiffuse(Material::FRONT_AND_BACK, color);
                    state->setAttribute(mat);
                    state->setRenderingHint(StateSet::TRANSPARENT_BIN);
                    state->setMode(GL_BLEND, StateAttribute::ON);
                    state->setMode(GL_LIGHTING, StateAttribute::OFF);
                    osg::PolygonMode* polymode = new osg::PolygonMode;
                    polymode->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::LINE);
                    state->setAttributeAndModes(polymode, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
                    fgeode->setStateSet(state);
                    // _annotations[inc]->geo = geo;
                    fgeode->addDrawable(geo);
                }

                float rotDegrees[3];
                rotDegrees[0] = 0;
                rotDegrees[1] = 0;
                rotDegrees[2] = 0;
                rotDegrees[0] = DegreesToRadians(rotDegrees[0]);
                rotDegrees[1] = DegreesToRadians(rotDegrees[1]);
                rotDegrees[2] = DegreesToRadians(rotDegrees[2]);
                Quat rot = osg::Quat(rotDegrees[0], osg::Vec3d(1, 0, 0), rotDegrees[1], osg::Vec3d(0, 1, 0), rotDegrees[2], osg::Vec3d(0, 0, 1));
                kinectFOV = new osg::MatrixTransform();
                Matrix rotMat;
                rotMat.makeRotate(rot);
                kinectFOV->setMatrix(rotMat);
                kinectFOV->addChild(fgeode);
                switchNode->addChild(kinectFOV.get());
	     }
            }

}
void KinectObject::loadTransformMenu()
{

    SubMenu* transforms_sm = new SubMenu("Load transform");
    addMenuItem(transforms_sm);
    transform_list = new std::vector<cvr::MenuButton*>();
    transform_path = new std::vector<string>();
    string directory = cvr::ConfigManager::getEntry("Plugin.KinectDemo.Transforms");
    DIR* dir;
    class dirent* ent;
    class stat st;
    dir = opendir(directory.c_str());

    while ((ent = readdir(dir)) != NULL) {
        const string file_name = ent->d_name;
        const string full_file_name = directory + "/" + file_name;

        if (file_name[0] == '.')
            continue;

        if (stat(full_file_name.c_str(), &st) == -1)
            continue;

        const bool is_directory = (st.st_mode & S_IFDIR) != 0;

        if (is_directory)
            continue;

        MenuButton* b = new MenuButton(file_name);//"test " + i);
        transforms_sm->addItem(b);
        b->setCallback(this);
        transform_list->push_back(b);
        transform_path->push_back(full_file_name);
    }

    closedir(dir);
}
