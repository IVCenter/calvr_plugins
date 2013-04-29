#include "Skeleton.h"

bool Skeleton::moveWithCam;
osg::Vec3d Skeleton::camPos;
osg::Vec3d Skeleton::camPos2;
osg::Quat Skeleton::camRot;
bool Skeleton::navSpheres;

JointNode::JointNode()
{
    translate = new osg::MatrixTransform();
    rotate    = new osg::MatrixTransform();
    geode     = new osg::Geode();
    rotate->addChild(geode);
    translate->addChild(rotate);
}

void JointNode::makeDrawable(int _id)
{
    id = _id;
    osg::Drawable* g;
    std::string color = "BL";

    switch (id)
    {
    case  1:
        color = "HE-HEAD";
        break;

    case  9:
        color = "GS-HAND";
        break;

    case 15:
        color = "GS-HAND";
        break;

    case 20:
        color = "FE-FOOT";
        break;

    case 24:
        color = "FE-FOOT";
        break;
    }

    osg::Box* sphereShape = new osg::Box(osg::Vec3d(0, 0, 0), 20);
    osg::ShapeDrawable* shapeDrawable = new osg::ShapeDrawable(sphereShape);
    shapeDrawable->setColor(getColor(color));
    g = shapeDrawable;
    geode->addDrawable(g);
}

void JointNode::update(int joint_id, float newx, float newy, float newz, float neworx, float newory, float neworz, float neworw, bool attached)
{
    // joint rotation
    orientation[0] = neworx;
    orientation[1] = newory;
    orientation[2] = neworz;
    orientation[3] = neworw;
    osg::Quat q = osg::Quat(neworx, newory, neworz, neworw);
    osg::Matrixd rmat;
    rmat.makeRotate(q);
    rotate->setMatrix(rmat);
    // joint position
    osg::Matrixd tmat;
    position.set(newx, newy, newz);
    position += Skeleton::camPos;
    tmat.makeTranslate(position);
    //camera rotation (probably doesn't work this way)
    osg::Matrix rot;
    rot.makeRotate(Skeleton::camRot);
    osg::Matrixd camSkel = rot * tmat;
    translate->setMatrix(camSkel);
}

NavigationSphere::NavigationSphere()
{
    position = osg::Vec3();
    prevPosition = osg::Vec3();
    osg::Vec3d poz0(0, 0, 0);
    osg::Sphere* sphereShape = new osg::Sphere(poz0, 0.1);
    osg::ShapeDrawable* ggg2 = new osg::ShapeDrawable(sphereShape);
    ggg2->setColor(osg::Vec4(0.1, 0.4, 0.3, 0.9));
    geode = new osg::Geode;
    geode->addDrawable(ggg2);
    rotate = new osg::MatrixTransform();
    osg::Matrix rotMat;
    rotMat.makeRotate(0, 1, 0, 1);
    rotate->setMatrix(rotMat);
    rotate->addChild(geode);
    translate = new osg::MatrixTransform();
    osg::Matrixd tmat;
    tmat.makeTranslate(poz0);
    translate->setMatrix(tmat);
    translate->addChild(rotate);
    lock = -1;
    activated = false;
}

void NavigationSphere::update(osg::Vec3d position2, osg::Vec4f orientation)
{
    osg::ShapeDrawable* newColor = (osg::ShapeDrawable*) geode->getDrawable(0);

    // XXX do this only when it actually changes (locks/unlocks)
    if (lock == -1)
    {
        newColor->setColor(osg::Vec4(0.3, 0.4, 0.2, 0.9));
    }
    else if (activated)
    {
        newColor->setColor(osg::Vec4(0.3, 0.4, 0.2, 0.9));
    }
    else
    {
        newColor->setColor(osg::Vec4(0.1, 0.4, 0.3, 0.9));
    }

    geode->setDrawable(0, newColor);
    position.set(position2);
    osg::Matrix rotMat;
    rotMat.makeRotate(orientation);
    rotate->setMatrix(rotMat);
    osg::Matrix posMat;
    posMat.makeTranslate(position2);
    translate->setMatrix(posMat);
}

Skeleton::Skeleton()
{
    rightHandBusy = false;
    leftHandBusy = false;

    //Algorithm for generating colors based on DC.
    if (colorsInitialized == false)
    {
        for (int i = 0; i < 729; i++)
        {
            _colors[i] = osg::Vec4(1 - float((i % 9) * 0.125), 1 - float(((i / 9) % 9) * 0.125), 1 - float(((i / 81) % 9) * 0.125), 1);
        }
    }

    for (int i = 0; i < 15; i++)
        bone[i] = MCylinder(10, osg::Vec4(0.3, 0.4, 0.2, 1.0));

    for (int i = 0; i < 25; i++)
    {
        joints[i].position.set(0, 0, 0);  // XXX does it matter? try removing
        joints[i].makeDrawable(i);
    }

    cylinder = MCylinder();
    navSphere = NavigationSphere();
    attached = false;
}

void Skeleton::update(int joint_id, float newx, float newy, float newz, float neworx, float newory, float neworz, float neworw)
{
    joints[joint_id].update(joint_id, newx, newy, newz, neworx, newory, neworz, neworw, attached);
    bone[0].update(joints[1].position, joints[2].position);
    bone[1].update(joints[2].position, joints[3].position);
    bone[3].update(joints[2].position, joints[6].position);
    bone[4].update(joints[2].position, joints[12].position);
    bone[2].update(joints[3].position, joints[17].position);
    bone[5].update(joints[3].position, joints[21].position);
    bone[6].update(joints[6].position, joints[7].position);
    bone[7].update(joints[7].position, joints[9].position);
    bone[9].update(joints[12].position, joints[13].position);
    bone[10].update(joints[13].position, joints[15].position);
    bone[11].update(joints[18].position, joints[20].position);
    bone[8].update(joints[22].position, joints[21].position);
    bone[12].update(joints[18].position, joints[17].position);
    bone[13].update(joints[22].position, joints[24].position);
}

//void Skeleton::attach(osg::MatrixTransform* parent)
void Skeleton::attach(osg::Switch* parent)
{
    attached = true;

    for (int i = 0; i < 15; i++)
        parent->addChild(bone[i].geode);

    for (int i = 0; i < 25; i++)
    {
        parent->addChild(joints[i].translate);
    }

    if (Skeleton::navSpheres) parent->addChild(navSphere.translate);
}

//void Skeleton::detach(osg::MatrixTransform* parent)
void Skeleton::detach(osg::Switch* parent)
{
    attached = false;

    for (int i = 0; i < 25; i++)
    {
        joints[i].translate->ref(); // XXX ugly hack
        parent->removeChild(joints[i].translate);
    }

    navSphere.translate->ref(); // XXX ugly hack
    parent->removeChild(navSphere.translate);
    parent->removeChild(cylinder.geode);

    for (int i = 0; i < 15; i++)    parent->removeChild(bone[i].geode);
}

osg::Vec4 JointNode::getColor(std::string dc)
{
    char letter1 = dc.c_str()[0];
    char letter2 = dc.c_str()[1];
    int char1 = letter1 - 65;
    int char2 = letter2 - 65;
    int tot = char1 * 26 + char2;
    return _colors[tot];
}
