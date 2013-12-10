#include "CloudManager.h"

#include <cvrKernel/ComController.h>
#include <sstream>
#include <algorithm>
#include <cstdio>

#include <sys/syscall.h>
#include <sys/stat.h>

#define VERTEXBIND 6

using namespace cvr;
using namespace std;

CloudManager::CloudManager(std::string server)
{
    useKColor = true;
    userColor = false;
    max_users = 12;
    radiusMin = 750;
    pause = false;
    should_quit = false;
    minDistHSV = 500;
    maxDistHSV = 4000;
    kinectVertices = new osg::Vec3Array;
    kinectNormals = new osg::Vec3Array;
    kinectColours = new osg::Vec4Array;
    _firstRun = 0;
    _next = true;
    kinectServer = server;

    // precomputing colors for heat coloring
    for (int i = 0; i < 20000; i++) getColorRGB(i);

    for (int i = 0; i < max_users; i++)
    {
        userSize.push_back(0);
        userOn.push_back(true);
        lastUserSize.push_back(0);
        lHandSize.push_back(0);
        lastlHandSize.push_back(0);
        userRadius.push_back(0);
    }

    // userOn[0] = false;
}

CloudManager::~CloudManager()
{
}

bool CloudManager::isCacheDone()
{
    return _next;
}
int CloudManager::firstRunStatus()
{
    return _firstRun;
}
void CloudManager::update()
{
    static bool frameLoading = false;
    bool cDone;
    /*
            if (cvr::ComController::instance()->isMaster())
            {
                cDone = _cacheDone;
                int numSlaves = cvr::ComController::instance()->getNumSlaves();
                bool sDone[numSlaves];
                cvr::ComController::instance()->readSlaves(sDone, sizeof(bool));

                for (int i = 0; i < numSlaves; i++)
                {
                    cDone = cDone && sDone[i];
                }

                cvr::ComController::instance()->sendSlaves(&cDone, sizeof(bool));
            }
            else
            {
                cDone = _cacheDone;
                cvr::ComController::instance()->sendMaster(&cDone, sizeof(bool));
                cvr::ComController::instance()->readMaster(&cDone, sizeof(bool));
            }

       cDone = _cacheDone;
           // run();

            if (!cDone)
            {
                //std::cerr << "Waiting for load to finish." << std::endl;
               // return;
            }

           // cvr::ComController::instance()->sync();
            //Add not here?
                //kinectVertices = newVertices;
                //kinectNormals = newNormals;
                //kinectColours = newColours;
       //     return;
            _next = true;
           // run();
        }

        if (cDone)
        {
            std::cerr << "Load Finished." << std::endl;
            //Add loaded node to root
        }
        else
        {
              std::cerr << "Waiting for GPU load finish." << std::endl;
        }
    */
}

void CloudManager::run()
{
    //Do functions
    _cacheDone = false;
    bool cDone = false;

    if (true)
    {
        //TODO:ZMQ does not want to be in init with the cloud socket-should only initialize this at the very beginning.
        zmq::context_t context2(1);
        cloudT_socket = new SubSocket<RemoteKinect::PointCloud> (context2, kinectServer);
        packet = new RemoteKinect::PointCloud();
        user1Vertices = new osg::Vec3Array(40000);
        user1Colours = new osg::Vec4Array(40000);
        newVertices = new osg::Vec3Array(0);
        newNormals = new osg::Vec3Array;
        newColours = new osg::Vec4Array(0);

        for (int i = 0; i < max_users; i++)
        {
            userVerticesArray.push_back(new osg::Vec3Array(1));
            userColoursArray.push_back(new osg::Vec4Array(1));
            drawArrays.push_back(new osg::DrawArrays(osg::PrimitiveSet::POINTS, 0, userVerticesArray[i].get()->size()));
            lHandVerticesArray.push_back(new osg::Vec3Array(1));
            lHandColoursArray.push_back(new osg::Vec4Array(1));
            drawArraysHand.push_back(new osg::DrawArrays(osg::PrimitiveSet::POINTS, 0, lHandVerticesArray[i].get()->size()));
        }

        while (!should_quit)
        {
            if (!_next)
            {
                // printf("X");
            }
            else
            {
                // printf("O");
                if (cloudT_socket != NULL)
                {
                    //printf("NotNull");
                    if (cloudT_socket->recv(*packet))
                    {
                        for (int n = 0; n < userSize.size(); n++)
                        {
                            userSize[n] = 0;
                        }

                        //Reset count for user Array Sizes
                        float r, g, b, a;
                        // overwrite ones that existed
                        osg::Vec4 tempColour = osg::Vec4(0, 0, 0, 1);

                        for (int i = 0; i < packet->points_size(); i++)
                        {
                            int id = packet->points(i).id();

                            if (userOn[id])
                            {
                                // userOn[0] == false if userColor
                                if (userColor == true && id == 0) continue;

                                osg::Vec3 ppos((packet->points(i).x()),
                                               (packet->points(i).z()),
                                               (packet->points(i).y()));

                                if (userColor)
                                {
                                    // color each user different color
                                    tempColour = getColorUser(packet->points(i).id(), packet->points(i).z());
                                }
                                else if (useKColor)
                                {
                                    r = (packet->points(i).r() / 255.);
                                    g = (packet->points(i).g() / 255.);
                                    b = (packet->points(i).b() / 255.);
                                    a = 1;
                                    tempColour = osg::Vec4(r, g, b, a);
                                }
                                else
                                {
                                    tempColour = getColorRGB(packet->points(i).z());
                                }

                                if (userSize[id] < userVerticesArray[id]->size())
                                {
                                    userVerticesArray[id]->at(userSize[id]) = ppos;
                                    userColoursArray[id]->at(userSize[id]) = tempColour;
                                }
                                else
                                {
                                    userVerticesArray[id]->push_back(ppos);
                                    userColoursArray[id]->push_back(tempColour);
                                }

                                userSize[id]++;
                            }
                        }

                        reduceUsersArray();
                        generateHandArray();

                        for (int n = 0; n < userVerticesArray.size(); n++)
                        {
                            if (userOn[n])
                            {
                                    drawArrays[n]->setCount(userSize[n]);
                                    userVerticesArray[n]->dirty();
                                    userColoursArray[n]->dirty();
                                    // userVerticesArray[n] = lHandVerticesArray[n];
                                    // userColoursArray[n] = lHandColoursArray[n];
                            }
                            else// if (userVerticesArray[n]->size() > 1)
                            {
                                //userVerticesArray[n]->resize(1);
                                //userColoursArray[n]->resize(1);
                                drawArrays[n]->setCount(0);//userVerticesArray[n]->size());
                                userVerticesArray[n]->dirty();
                                userColoursArray[n]->dirty();
                            }
                        }

                        kinectVertices = userVerticesArray[0];
                        kinectColours = userColoursArray[0];
                    }
                    else
                    {
                        //cerr << "PointCloud " << kinectServer << " Empty\n";
                    }
                }

                _next = false;
                _cacheDone = true;
            }

            //This is the initial Function that runs any type of processing on Kinect PointCloud sets before overwriting old!
            //processNewCloud();

            if (_firstRun == 1)
            {
                _firstRun = 2;
            }
            else if (_firstRun == 0)
            {
                _firstRun = 1;
            }

            _cacheDone = false;
            _next = true;
        }
    }

    delete cloudT_socket;
    cloudT_socket = NULL;
    //When Finished
    std::cerr << "All frames loaded." << std::endl;
}

void  CloudManager::quit()
{
    should_quit = true;
}

osg::Vec4f CloudManager::getColorRGB(int dist)
{
    if (distanceColorMap.count(dist) == 0) // that can be commented out after precomputing completely if the range of Z is known (and it is set on the server side)
    {
        float r, g, b;
        float h = depth_to_hue(minDistHSV, dist, maxDistHSV);
        HSVtoRGB(&r, &g, &b, h, 1, 1);
        distanceColorMap[dist] = osg::Vec4f(r, g, b, 1);
    }

    return distanceColorMap[dist];
}

osg::Vec4f CloudManager::getColorUser(int uid, int dist)
{
    if (userColorMap.count(uid * 1000000 + dist) == 0)
    {
        float r, g, b, h, s, v;
        h = (360.0 / max_users) * (uid - 1) / 360.0;
        s = (float)(dist - minDistHSV) / (float)maxDistHSV;

        if (s < 0) s = 0;

        if (s > 1) s = 1;

        v = 1.0;
        //cout << "uid " << uid << " at " << dist << " : h " << h << " s " << s << " v " << v << endl;
        HSVtoRGB(&r, &g, &b, h * 360, s, v);
        userColorMap[uid * 1000000 + dist] = osg::Vec4f(r, g, b, 1);
    }

    return userColorMap[uid * 1000000 + dist];
    /*

    if (uid == 0) return osg::Vec4f(0, 0, 0, 1);
    else if (uid == 1) return osg::Vec4f(1, 0, 0, 1);
    else if (uid == 2) return osg::Vec4f(0, 1, 0, 1);
    else if (uid == 3) return osg::Vec4f(0, 0, 1, 1);
    else return osg::Vec4f(1, 0, 1, 1);

    if (userColorMap.count(uid) == 0)
    {
        float r, g, b;
        dist = uid * 1000;
        float h = depth_to_hue(minDistHSV, dist, maxDistHSV);
        HSVtoRGB(&r, &g, &b, h, 1, 1);
        userColorMap[dist] = osg::Vec4f(r, g, b, 1);
    }

    return userColorMap[dist];*/
}
void CloudManager::processNewCloud()
{
    //Here is where we could run ICP or do checking against original cloud.
}
void CloudManager::reduceUsersArray()
{
    for (int n = 0; n < userSize.size(); n++)
    {
        int diff = -userSize[n] + userVerticesArray[n]->size();

        if (diff > 0)
        {
            while (diff-- > 0)
            {
                userVerticesArray[n]->pop_back();
                userColoursArray[n]->pop_back();
            }
        }
    }

    for (int n = 0; n < userSize.size(); n++)
    {
        if (userOn[n] == 0)
        {
            int diff = userVerticesArray[n]->size();

            if (diff > 0)
            {
                while (diff-- > 0)
                {
                    userVerticesArray[n]->pop_back();
                    userColoursArray[n]->pop_back();
                }
            }
        }
    }
}
void CloudManager::reduceHandArray()
{
    for (int n = 0; n < lHandSize.size(); n++)
    {
        if (lHandSize[n] < lastlHandSize[n] && lastlHandSize[n] != 0 && lHandSize[n] != 0)
        {
            int removals = lastlHandSize[n] - lHandSize[n];

            while (removals > 0)
            {
                lHandVerticesArray[n]->pop_back();
                lHandColoursArray[n]->pop_back();
                removals--;
            }
        }

        lastlHandSize[n] = lHandSize[n];
    }
}
void CloudManager::generateHandArray()
{
    for (int i = 1; i < userVerticesArray.size(); i++)
    {
        if (userVerticesArray[i]->size() > 500)
        {
            osg::Geometry* tempGeom = new osg::Geometry();
            osg::DrawArrays* drawArray = new osg::DrawArrays(osg::PrimitiveSet::POINTS, 0, userVerticesArray[i]->size());
            tempGeom->addPrimitiveSet(drawArray);
            tempGeom->setVertexArray(userVerticesArray[i]);
            //Get New Bounding Box and Compare to Average
            tempGeom->computeBound();
            osg::BoundingBox bbox = tempGeom->getBound();
            osg::Vec3f center = bbox.center();
            float yOverRadius = center.y() - 225;
            float yOverLimit = center.y() - 800 ;
            int id = i;
            lHandSize[id] = 0;

            for (int n = 0; n < userVerticesArray[id]->size(); n++)
            {
                if (userVerticesArray[id]->at(n).y() < yOverRadius && userVerticesArray[id]->at(n).y() > yOverLimit)
                {
                    if (lHandSize[id] < lHandVerticesArray[id]->size())
                    {
                        lHandVerticesArray[id]->at(lHandSize[id]) = userVerticesArray[id]->at(n);
                        lHandColoursArray[id]->at(lHandSize[id]) = userColoursArray[id]->at(n);
                    }
                    else
                    {
                        lHandVerticesArray[id]->push_back(userVerticesArray[id]->at(n));
                        lHandColoursArray[id]->push_back(userColoursArray[id]->at(n));
                    }

                    lHandSize[id]++;
                    //TODO: Need to do Cluster to find which hand it belongs too
                }
            }

            reduceHandArray();
            drawArraysHand[i]->setCount(lHandVerticesArray[i]->size());
            lHandVerticesArray[i]->dirty();
            lHandColoursArray[i]->dirty();
        }
        else
        {
            if (lHandVerticesArray[i]->size() > 1)
            {
                lHandVerticesArray[i]->resize(1);
                lHandColoursArray[i]->resize(1);
                drawArraysHand[i]->setCount(lHandVerticesArray[i]->size());
                lHandVerticesArray[i]->dirty();
                lHandColoursArray[i]->dirty();
            }
        }
    }
}
void CloudManager::triangulatePointCloud()
{
    // create a square area
    // osg::Vec3Array* points = new osg::Vec3Array;
    // points->push_back(osg::Vec3(-1, -1, 0));
    // points->push_back(osg::Vec3(-1,  1, 0));
    // points->push_back(osg::Vec3( 1, -1, 0));
    // points->push_back(osg::Vec3( 1,  1, 0));
    //
    // // create triangulator and set the points as the area
    // osg::ref_ptr<osgUtil::DelaunayTriangulator> trig = new osgUtil::DelaunayTriangulator();
    // trig->setInputPointArray(points);
    //
    // // create a triangular constraint
    // osg::Vec3Array* bounds = new osg::Vec3Array;
    // bounds->push_back(osg::Vec3(-0.5f, -0.5f, 0));
    // bounds->push_back(osg::Vec3(-0.5f, 0.5f, 0));
    // bounds->push_back(osg::Vec3(0.5f, 0.5f, 0));
    // osg::ref_ptr<osgUtil::DelaunayConstraint> constraint = new osgUtil::DelaunayConstraint;
    // constraint->setVertexArray(bounds);
    // constraint->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINE_LOOP,0,3) );
    //
    // // add constraint to triangulator
    // trig->addInputConstraint(constraint.get());
    //
    // trig->triangulate();
    //
    // // remove triangle from mesh
    // trig->removeInternalTriangles(constraint.get());
    //
    // // put triangulated mesh into OSG geometry
    // osg::Geometry* gm = new osg::Geometry;
    // gm->setVertexArray(points);
    // gm->addPrimitiveSet(trig->getTriangles());
    // osg::Vec4Array* colors = new osg::Vec4Array(1);
    // colors->push_back(osg::Vec4(1,0,1,1));
    // gm->setColorArray(colors);
    // gm->setColorBinding(osg::Geometry::BIND_OVERALL);
    //
}
void CloudManager::updateUserVertices(int id, int j, osg::Vec3 ppos)
{
    if (id == 1)
    {
        if (userVerticesArray[id]->size() > j)
        {
            userVerticesArray[id]->at(j) = ppos;

            if (id == 1)
            {
                user1Vertices->at(j) = ppos;
            }
        }
        else
        {
            userVerticesArray[id]->push_back(ppos);

            if (id == 1)
            {
                user1Vertices->push_back(ppos);
            }
        }
    }
}

void CloudManager::updateHandVertices(int id, int j, int hand, osg::Vec3 ppos)
{
    if (lHandVerticesArray[id]->size() > j)
    {
        lHandVerticesArray[id]->at(j) = ppos;
    }
    else
    {
        lHandVerticesArray[id]->push_back(ppos);
    }
}
void CloudManager::updateUserColours(int id, int j, osg::Vec4 tempColour)
{
    if (id == 1)
    {
        if (userColoursArray[id]->size() > j)
        {
            userColoursArray[id]->at(j) = tempColour;

            if (id == 1)
            {
                user1Colours->at(j) = tempColour;
            }
        }
        else
        {
            userColoursArray[id]->push_back(tempColour);

            if (id == 1)
            {
                user1Colours->push_back(tempColour);
            }
        }
    }
}
void CloudManager::updateHandColours(int id, int j, int hand, osg::Vec4 tempColour)
{
    if (lHandColoursArray[id]->size() > j)
    {
        lHandColoursArray[id]->at(j) = tempColour;
    }
    else
    {
        lHandColoursArray[id]->push_back(tempColour);
    }
}
