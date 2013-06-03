
#include "AudioHandler.h"


AudioHandler::AudioHandler()
{
    _isConnected = false;

    std::string name = "Server name";
    std::string dir = "/home/ryiah/Work/data/";

    uint8_t packet[256];
    int32_t size;
    size = oscpack(packet, "/sc.elevators/mastervol", "f", -1.0);
    send(_sock, (char *) packet, size, 0);

//    ColliderPlusPlus::Client_Server * server = 
//        new ColliderPlusPlus::Client_Server(name, "127.0.0.1", "1234", dir);
}

AudioHandler::~AudioHandler()
{
    if (_sock) 
    {
        close(_sock);
    }
}

void AudioHandler::connectServer()
{
    // check with configuration file to decide use audio server or not
    std::string useASStr = cvr::ConfigManager::getEntry("Plugin.ElevatorRoom.AudioServer");
    if (useASStr.compare("on")) 
        return;

    // get server address and port number
    std::string server_addr = cvr::ConfigManager::getEntry("Plugin.ElevatorRoom.AudioServerAddress");
    int port_number = cvr::ConfigManager::getInt("Plugin.ElevatorRoom.AudioServerPort", 15003);

    if( server_addr == "" ) 
        server_addr = "127.0.0.1";

    // build up socket communication
    int portno = port_number, protocol = SOCK_DGRAM;
    struct sockaddr_in server;
    struct hostent *hp;

    _sock = socket(AF_INET, protocol, 0);
    if (_sock < 0)
    {
        std::cerr << "ElevatorRoom::connectServer(): Can't open socket for audio server." << std::endl;
        return;
    }

    server.sin_family = AF_INET;
    hp = gethostbyname(server_addr.c_str());
    if (hp == 0)
    {
        std::cerr << "ElevatorRoom::connectServer(): Unknown audio server host." << std::endl;
        close(_sock);
        return;
    }
    memcpy((char *)&server.sin_addr, (char *)hp->h_addr, hp->h_length);
    server.sin_port = htons((unsigned short)portno);

    // connect to audio server
    if (connect(_sock, (struct sockaddr *) &server, sizeof (server)) < 0)
    {
        std::cerr << "ElevatorRoom::connect(): Failed connect to server" << std::endl;
        close(_sock);
        return;
    }

    std::cerr << "ElevatorRoom::AudioConfigHandler::Successfully connected to Audio Server." << std::endl;

    _isConnected = true;
}

void AudioHandler::disconnectServer()
{
    close(_sock);
    _isConnected = false;
    _sock = 0;
}

void AudioHandler::loadSound(int soundID, osg::Vec3 &dir, osg::Vec3 &pos)
{
    if (!_isConnected)
        return;

    uint8_t packet[256];
    int32_t size;

    float yaw, pitch, roll;

/*    yaw = acos(dir.x());
    if (dir.y() < 0) 
        yaw = M_PI * 2 - yaw;
    yaw += M_PI * 0.5;
    if (yaw > M_PI * 2) 
        yaw -= M_PI * 2;*/

    float val;
    if(dir.x() >= 0)
    {
        if(dir.y() >= 0)
        {
            val = acos( dir.x() );
        }
        else
        {
            val = 2*M_PI - acos( dir.x() );
        }
    }
    else
    {
        if(dir.y() >= 0)
        {
            val = M_PI - acos( -dir.x() );
        }
        else
        {
            val = M_PI + acos( -dir.x() );
        }
    }
    yaw = val;

    pitch = M_PI * 0.5 - atan(dir.z());
    roll = 0.0f;

    // yaw in DEGREES
    yaw = 0; 
    std::cout << soundID << std::endl;
    size = oscpack(packet, "/sc.elevators/poseaz", "if", soundID, yaw);

//    size = oscpack(packet, "/sc.elevators/pose", "iffffff", soundID, 
//        pos.x() / 1000, pos.y() / 1000, pos.z() / 1000, yaw, pitch, roll);

    if (send(_sock, (char *) packet, size, 0))
    {
        //std::cerr << "AudioHandler  Viewer's info: " << viewPos.x() << " " << viewPos.y() << " " << viewPos.z() << std::endl;	
    }
}

void AudioHandler::loadSound(int soundID, float az)
{
    if (!_isConnected)
        return;

    uint8_t packet[256];
    int32_t size;

    size = oscpack(packet, "/sc.elevators/poseaz", "if", soundID, az);

    send(_sock, (char *) packet, size, 0);
}

void AudioHandler::playSound(int soundID, std::string sound)
{
    if (!_isConnected)
        return;

    uint8_t packet[256];
    int32_t size;

    size = oscpack(packet, "/sc.elevators/trigger", "is", soundID, sound.c_str());
    send(_sock, (char *) packet, size, 0);
}

void AudioHandler::update(int soundID, const osg::Vec3 &dir, const osg::Vec3 &pos)
{
    if (!_isConnected)
        return;

    uint8_t packet[256];
    int32_t size;

    // OSCPack Messenger: send pose info within every frame, typ = 0 for observer
    float yaw, pitch, roll;

    /*yaw = acos(dir.x());
    if (dir.y() < 0) 
        yaw = M_PI * 2 - yaw;
    yaw += M_PI * 0.5;
    if (yaw > M_PI * 2) 
        yaw -= M_PI * 2;*/

    float val;
    if(dir.x() >= 0)
    {
        if(dir.y() >= 0)
        {
            val = acos( dir.x() );
        }
        else
        {
            val = 2*M_PI - acos( dir.x() );
        }
    }
    else
    {
        if(dir.y() >= 0)
        {
            val = M_PI - acos( -dir.x() );
        }
        else
        {
            val = M_PI + acos( -dir.x() );
        }
    }
    yaw = val;

    pitch = M_PI * 0.5 - atan(dir.z());
    roll = 0.0f;
    

    // yaw in DEGREES
    yaw = 0;
    size = oscpack(packet, "/sc.elevators/poseaz", "if", soundID, yaw);

//  size = oscpack(packet, "/sc.elevators/pose", "iffffff", soundID, 
//        pos.x() / 1000, pos.y() / 1000, pos.z() / 1000, yaw, pitch, roll);
    if (send(_sock, (char *) packet, size, 0))
    {
        //std::cerr << "AudioHandler  Viewer's info: " << viewPos.x() << " " << viewPos.y() << " " << viewPos.z() << std::endl;	
    }
}

void AudioHandler::update(int soundID, float az)
{
    if (!_isConnected)
        return;

    uint8_t packet[256];
    int32_t size;

    size = oscpack(packet, "/sc.elevators/poseaz", "if", soundID, az);

    send(_sock, (char *) packet, size, 0);
}
