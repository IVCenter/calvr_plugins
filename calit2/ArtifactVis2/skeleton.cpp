#include "skeleton.h"
#include <osg/Material>
#include <osg/MatrixTransform>
#include <osgText/Text>
#include <osg/Vec4>

#include <osgUtil/SceneView>
#include <osg/Camera>

#include <osg/CullFace>
#include <osg/Matrix>
#include <osg/ShapeDrawable>
#include <osgDB/ReadFile>
#include <osg/PositionAttitudeTransform>
#include <osg/PolygonMode>

const double m2mm = 1000.0;

bool Skeleton::moveWithCam;
osg::Vec3d Skeleton::camPos;
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
    osg::Vec3d zero(0, 0, 0);
    // TODO tessellation - ?
    int tessellation = 0;
    std::string color = "BL";

    if (id == 1) color = "HEAD";

    int sizemlt = 1;

    if (id == 9 || id == 15) {
        color = "GS-HAND";
        sizemlt = 2;
    }

    if (id == 20 || id == 24) color = "FE-FOOT";

    if (id == 3)
    {
        color = "FE-HAND";
    }
    //double _sphereRadius = m2mm * 0.03;
    //osg::Box* sphereShape = new osg::Box(zero, _sphereRadius * sizemlt);
    double _sphereRadius = 0.03;
    osg::Box* sphereShape = new osg::Box(zero, _sphereRadius);
    osg::ShapeDrawable* shapeDrawable = new osg::ShapeDrawable(sphereShape);
    shapeDrawable->setColor(getColor(color));
    g = shapeDrawable;
    geode->addDrawable(g);
}

void JointNode::update(int joint_id, osg::Vec3d pos, double ori[4], bool attached, bool lHandOpen, bool rHandOpen)
{
    double x = pos.x();
    double y = pos.y();
    double z = pos.z();
    position.set(x, y, z);

    for (int i = 0; i < 4; i++)
    {
        if (ori[i] != -1) orientation[i] = ori[i];
    }

    osg::Quat q = osg::Quat(ori[0], ori[1], ori[2], ori[3]);
    osg::Matrixd rmat;
    rmat.makeRotate(q);
    rotate->setMatrix(rmat);
    osg::Matrixd tmat;
    osg::Vec3d posDraw = position;
    //if (Skeleton::moveWithCam) // if uncommented, skeleton will return to raw coordinates after option is turned off
    position += Skeleton::camPos;
    tmat.makeTranslate(position);
    //tmat.makeTranslate(Skeleton::camPos);
    osg::Matrix rot;
    rot.makeRotate(Skeleton::camRot);
    osg::Matrixd camSkel = rot * tmat;

    translate->setMatrix(camSkel);
    //osition = camSkel.getTrans();
    //osg::Matrixd tmat2;

    //tmat2 = rot * camSkel;
    ////rot.setPosition(position);
    //translate->setMatrix(tmat2);
    if(id == 1)
    {
        //osg::Quat rott = Skeleton::camRot;
        //printf("%g,%g,%g,%g\n",rott.x(),rott.y(),rott.z(),rott.w());
    }
    if (id == 9 && lHandOpen)
    {
        osg::ShapeDrawable* shDr = (osg::ShapeDrawable*) geode->getDrawable(0);
        shDr->setColor(getColor("BL"));
    }

    if (id == 9 && !lHandOpen)
    {
        osg::ShapeDrawable* shDr = (osg::ShapeDrawable*) geode->getDrawable(0);
        shDr->setColor(getColor("FE"));
    }

    if (id == 15 && rHandOpen)
    {
        osg::ShapeDrawable* shDr = (osg::ShapeDrawable*) geode->getDrawable(0);
        shDr->setColor(getColor("BL"));
    }

    if (id == 15 && !rHandOpen)
    {
        osg::ShapeDrawable* shDr = (osg::ShapeDrawable*) geode->getDrawable(0);
        shDr->setColor(getColor("FE"));
    }
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
    if(lock == -1)
    {
        newColor->setColor(osg::Vec4(0.3, 0.4, 0.2, 0.9));
    }
    else if(activated)
    {

        newColor->setColor(osg::Vec4(0.3, 0.4, 0.2, 0.9));
    }
    else
    {
        newColor->setColor(osg::Vec4(0.1, 0.4, 0.3, 0.9));

    }

    geode->setDrawable(0,newColor);
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
    YRES = 480;
    XRES = 640;
    ROIOFFSET = 90; // originally 70
    DEPTH_SCALE_FACTOR = 255. / 4096.;
    BIN_THRESH_OFFSET = 9;
    MEDIAN_BLUR_K = 5;
    GRASPING_THRESH = 0.9;
    HAND_SIZE = 500; // originally 2000
    lHandOpen = false;
    rHandOpen = false;

    if (colorsInitialized == 0) {
        colorsInitialized = 1;

        for (int i = 0; i < 729; i++)
        {
            _colorsJoints[i] = osg::Vec4(1 - float((i % 9) * 0.125), 1 - float(((i / 9) % 9) * 0.125), 1 - float(((i / 81) % 9) * 0.125), 1);
        }
    }

    for (int i = 0; i < 15; i++)
        bone[i] = MCylinder(0.003, osg::Vec4(0.3, 0.4, 0.2, 1.0));

    for (int i = 0; i < 25; i++)
    {
        joints[i].position.set(0, 0, 0);  // TODO does it matter?
        joints[i].makeDrawable(i);
    }

    cylinder = MCylinder();
    navSphere = NavigationSphere();
    attached = false;
    offset = osg::Vec3(0, 0, 0);
    //roi.width = ROI_OFFSET * 2;
    //roi.height = ROI_OFFSET * 2;
}

void Skeleton::addOffset(osg::Vec3 off)
{
    offset.set(offset.x() + off.x(), offset.y() + off.y(), offset.z() + off.z());
}

void Skeleton::update(int joint_id, osg::Vec3d pos, double ori[4])
{
    osg::Vec3 pos2 = osg::Vec3(pos.x() + offset.x(), pos.y() + offset.y(), pos.z() + offset.z());
    joints[joint_id].update(joint_id, pos2, ori, attached, lHandOpen, rHandOpen);

    // draw 'bones'
    if (attached)
    {
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
}

void Skeleton::attach(osg::MatrixTransform* parent)
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

void Skeleton::detach(osg::MatrixTransform* parent)
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
    // TODO could be merged with the one in artifactvis2
    char letter1 = dc.c_str() [0];
    char letter2 = dc.c_str() [1];
    int char1 = letter1 - 65;
    int char2 = letter2 - 65;
    int tot = char1 * 26 + char2;
    return _colorsJoints[tot];
}

// conversion from cvConvexityDefect
/*
// Thanks to Jose Manuel Cabrera for part of this C++ wrapper function
void Skeleton::findConvexityDefects(std::vector<cv::Point>& contour, std::vector<int>& hull, std::vector<ConvexityDefect>& convexDefects)
{

    if (hull.size() > 0 && contour.size() > 0)
    {
        CvSeq* contourPoints;
        CvSeq* defects;
        CvMemStorage* storage;
        CvMemStorage* strDefects;
        CvMemStorage* contourStr;
        CvConvexityDefect* defectArray = 0;
        strDefects = cvCreateMemStorage();
        defects = cvCreateSeq(CV_SEQ_KIND_GENERIC | CV_32SC2, sizeof(CvSeq), sizeof(CvPoint), strDefects);
        //We transform our vector<Point> into a CvSeq* object of CvPoint.
        contourStr = cvCreateMemStorage();
        contourPoints = cvCreateSeq(CV_SEQ_KIND_GENERIC | CV_32SC2, sizeof(CvSeq), sizeof(CvPoint), contourStr);

        for (int i = 0; i < (int) contour.size(); i++) {
            CvPoint cp = {contour[i].x,  contour[i].y};
            cvSeqPush(contourPoints, &cp);
        }

        //Now, we do the same thing with the hull index
        int count = (int) hull.size();
        //int hullK[count];
        int* hullK = (int*) malloc(count * sizeof(int));

        for (int i = 0; i < count; i++) {
            hullK[i] = hull.at(i);
        }

        CvMat hullMat = cvMat(1, count, CV_32SC1, hullK);
        // calculate convexity defects
        storage = cvCreateMemStorage(0);
        defects = cvConvexityDefects(contourPoints, &hullMat, storage);
        defectArray = (CvConvexityDefect*) malloc(sizeof(CvConvexityDefect) * defects->total);
        cvCvtSeqToArray(defects, defectArray, CV_WHOLE_SEQ);
        //printf("DefectArray %i %i\n",defectArray->end->x, defectArray->end->y);

        //We store defects points in the convexDefects parameter.
        for (int i = 0; i < defects->total; i++) {
            ConvexityDefect def;
            def.start       = cv::Point(defectArray[i].start->x, defectArray[i].start->y);
            def.end         = cv::Point(defectArray[i].end->x, defectArray[i].end->y);
            def.depth_point = cv::Point(defectArray[i].depth_point->x, defectArray[i].depth_point->y);
            def.depth       = defectArray[i].depth;
            convexDefects.push_back(def);
        }

        // release memory
        cvReleaseMemStorage(&contourStr);
        cvReleaseMemStorage(&strDefects);
        cvReleaseMemStorage(&storage);
    }

}

void Skeleton::checkHandOpen(int handId, cv::Mat handMat, int handDepth, int _ROIOFFSET)
{

    ROIOFFSET = _ROIOFFSET;
    // 70 - 2000
    // 40 - 500
    HAND_SIZE = (ROIOFFSET * 0.4 * ROIOFFSET);
    handMat = (handMat > (handDepth - BIN_THRESH_OFFSET)) & (handMat < (handDepth + BIN_THRESH_OFFSET));
    cv::medianBlur(handMat, handMat, MEDIAN_BLUR_K);
    std::vector< std::vector<cv::Point> > contours;
    cv::findContours(handMat, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

    if (contours.size()) {
        for (int i = 0; i < contours.size(); i++) {
            std::vector<cv::Point> contour = contours[i];
            cv::Mat contourMat = cv::Mat(contour);
            double cArea = cv::contourArea(contourMat);

            if (cArea > HAND_SIZE) // likely the hand
            {
                printf("saw a hand\n");
                cv::Scalar center = cv::mean(contourMat);
                cv::Point centerPoint = cv::Point(center.val[0], center.val[1]);
                // approximate the contour by a simple curve
                std::vector<cv::Point> approxCurve;
                cv::approxPolyDP(contourMat, approxCurve, 10, true);
                std::vector< std::vector<cv::Point> > debugContourV;
                std::vector<int> hull;
                cv::convexHull(cv::Mat(approxCurve), hull, false, false);
                // find convexity defects
                std::vector<ConvexityDefect> convexDefects;
                findConvexityDefects(approxCurve, hull, convexDefects);
                printf("Number of defects: %d.\n", (int) convexDefects.size());
                // assemble point set of convex hull
                std::vector<cv::Point> hullPoints;

                for (int k = 0; k < hull.size(); k++)
                {
                    int curveIndex = hull[k];
                    cv::Point p = approxCurve[curveIndex];
                    hullPoints.push_back(p);
                }

                // area of hull and curve
                double hullArea  = cv::contourArea(cv::Mat(hullPoints));
                double curveArea = cv::contourArea(cv::Mat(approxCurve));
                double handRatio = curveArea / hullArea;

                // hand is grasping
                if (handRatio > GRASPING_THRESH)
                {   //circle(debugFrames[handI], centerPoint, 5, COLOR_LIGHT_GREEN, 5);
                    printf("closed hand\n");

                    if (handId == 9) lHandOpen = false;

                    if (handId == 15) rHandOpen = false;
                }
                else
                {   //circle(debugFrames[handI], centerPoint, 5, COLOR_RED, 5);
                    printf("open hand\n");

                    if (handId == 9) lHandOpen = true;

                    if (handId == 15) rHandOpen = true;
                }
            }
        } // contour conditional
    } // hands loop
}
*/
