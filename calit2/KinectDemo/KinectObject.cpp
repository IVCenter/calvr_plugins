#include "KinectObject.h"
#define VERTEXBIND 6

using namespace cvr;
using namespace std;
using namespace osg;

KinectObject::KinectObject(std::string name, std::string cloud_server, std::string skeleton_server, osg::Vec3 position) : SceneObject(name, false, false, false, true, false)
{
    //Setup Extended SceneObject
    switchNode = new osg::Switch();
    addChild(switchNode);
    setNavigationOn(true);
    setMovable(false);
    addMoveMenuItem();
    addNavigationMenuItem();
    //   kinectX = ConfigManager::getFloat("x", "Plugin.KinectDemo.KinectSkeleton", 0.0f);
    //   kinectY = ConfigManager::getFloat("y", "Plugin.KinectDemo.KinectSkeleton", 0.0f);
    //   kinectZ = ConfigManager::getFloat("z", "Plugin.KinectDemo.KinectSkeleton", 0.0f);
    kinectX = position.x();
    kinectY = position.y();
    kinectZ = position.z();
    cloudServer = cloud_server;
    skeletonServer = skeleton_server;
    setPosition(position);
    //Setup Cloud
    cloudInit();
    skeletonOn();

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
        cm = new CloudManager(cloudServer);
        cout << "Started\n";
        cm->start();
        group = new osg::Group();
        osg::StateSet* state = group->getOrCreateStateSet();
        state->setAttribute(pgm1);
        state->addUniform(new osg::Uniform("pointScale", initialPointScale));
        state->addUniform(new osg::Uniform("globalAlpha", 1.0f));
        float pscale = initialPointScale;
        osg::Uniform*  _scaleUni = new osg::Uniform("pointScale", 1.0f * pscale);
        group->getOrCreateStateSet()->addUniform(_scaleUni);
        switchNode->addChild(group);
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
    if (cm != NULL)
    {
        _cloudIsOn = false;
        cm->quit();
        group->removeChild(0, 1);
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
                        kgeode = new osg::Geode();
                        kgeode->setCullingActive(false);
                        geom = new osg::Geometry();
                        StateSet* state = geom->getOrCreateStateSet();
                        state->setMode(GL_LIGHTING, StateAttribute::OFF | osg::StateAttribute::OVERRIDE);
                        osg::VertexBufferObject* vboP = geom->getOrCreateVertexBufferObject();
                        vboP->setUsage(GL_STREAM_DRAW);
                        //vboP->setUsage(GL_DYNAMIC_DRAW);
                        //vboP->setUsage(GL_STATIC_DRAW);
                        geom->setUseDisplayList(false);
                        geom->setUseVertexBufferObjects(true);
                        kinectVertices = new Vec3Array;
                        kinectColours = new Vec4Array;
                        drawArray = new osg::DrawArrays(osg::PrimitiveSet::POINTS, 0, cm->kinectVertices.get()->size());
                        geom->addPrimitiveSet(drawArray);

                        //geom->setVertexArray(cm->kinectVertices.get());
                        geom->setVertexAttribArray(VERTEXBIND, cm->kinectVertices.get());
                        geom->setVertexAttribBinding(VERTEXBIND, osg::Geometry::BIND_PER_VERTEX);

                        geom->setColorArray(cm->kinectColours.get());
                        geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
                        kgeode->addDrawable(geom);
                        group->removeChild(0, 1);
                        group->addChild(kgeode);
                    }
                    else
                    {
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
        if(skel_frame->skeletons_size() > 0)
      //  cerr << "Skels:" << skel_frame->skeletons_size() << "\n";
        // update all skeletons' joints' positions
        for (int i = 0; i < skel_frame->skeletons_size(); i++)
        {
            // Skeleton reported but not in the map -> create a new one
            if (mapIdSkel.count(skel_frame->skeletons(i).skeleton_id()) == 0)
            {
                mapIdSkel[skel_frame->skeletons(i).skeleton_id()] = Skeleton(); ///XXX remove Skeleton(); part
                // mapIdSkel[sf->skeletons(i).skeleton_id()].attach(_root);
                cerr << "Found Skeleton\n";
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

std::map<int, Skeleton>* KinectObject::skeletonGetMap()
{
    return &mapIdSkel;
}
