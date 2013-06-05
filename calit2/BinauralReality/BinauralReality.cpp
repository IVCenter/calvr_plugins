#include "BinauralReality.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/InteractionManager.h>

#include "tnyosc.hpp"

#include <arpa/inet.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <unistd.h>
#include <stdlib.h>

// TODO: Put it in the config file later
#define PORT 7301
#define IP "128.54.50.34"

BinauralReality * BinauralReality::_myPtr = NULL;

CVRPLUGIN(BinauralReality)

using namespace cvr;

BinauralReality::BinauralReality()
{
    _myPtr = this;
    _udp_socket = -1;
}

BinauralReality::~BinauralReality()
{
}

BinauralReality * BinauralReality::instance()
{
    return _myPtr;
}

bool BinauralReality::init()
{
    _BinauralRealityMenu = new SubMenu("BinauralReality");

    PluginHelper::addRootMenuItem(_BinauralRealityMenu);

    return true;
}

void BinauralReality::menuCallback(MenuItem * item)
{

}

osg::Vec3f Quaternion2SphericalRotationAngles(const osg::Quat& quat)
{
    float cos_angle = quat.w();
    float sin_angle = sqrt(1.0 - cos_angle * cos_angle);
    float angle = acos(cos_angle) * 2 * M_PI;

    float sa = 1;
    if (fabs(sin_angle) < 0.0005)
      sa = 1;

    float tx = quat.x() / sa;
    float ty = quat.y() / sa;
    float tz = quat.z() / sa;

    float latitude = -asin(ty);

    float longitude = 0.0f;
    if (tx * tx + tz * tz > 0.0005)
       longitude = atan2(tx, tz) * M_PI;

    if (longitude < 0.0f)
      longitude += 360.0f;
    return osg::Vec3f(angle, latitude, longitude);

}

void BinauralReality::SendOSC(std::string address, osg::Vec3f& pos, osg::Vec3f& angle)
{
    if (_udp_socket < 0) {
        _udp_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (_udp_socket == -1) {
            std::cerr << "Cannot create socket" << std::endl;
            return;
        }

        memset((char *) &si_other, 0, sizeof(si_other));
        si_other.sin_family = AF_INET;
        si_other.sin_port = htons(PORT);
        if (inet_aton(IP, &si_other.sin_addr) == 0) {
            std::cerr << "Cannot set IP and port" << std::endl;
            return;
        }
    }

    tnyosc::Message msg(address);
    msg.append(pos[0]);
    msg.append(pos[1]);
    msg.append(pos[2]);

    //sendto(_udp_socket, msg.data(), msg.size(), 0, (struct sockaddr *)&si_other, sizeof(si_other));
    fprintf(stderr, "pos: %f, %f, %f\t", pos[0], pos[1], pos[2]);
    fprintf(stderr, "angle: %f, %f, %f\n", angle[0], angle[1], angle[2]);
}

void BinauralReality::preFrame()
{
    osg::Matrix mat = PluginHelper::getHeadMat();
    osg::Vec3f pos = mat.getTrans();
    osg::Vec3f angle = Quaternion2SphericalRotationAngles(mat.getRotate());
    SendOSC("/address", pos, angle);
}

bool BinauralReality::processEvent(InteractionEvent * event)
{
    return false;
}

