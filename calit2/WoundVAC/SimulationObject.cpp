#include "SimulationObject.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/PluginHelper.h>

#include <osg/ShapeDrawable>

#include <iostream>
#include <sstream>
#include <sys/select.h>
#include <time.h>

#include <jsoncpp/json/reader.h>
#include <jsoncpp/json/writer.h>

using namespace cvr;

SimulationObject::SimulationObject(cvr::CVRSocket * socket, std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds) : SceneObject(name,navigation,movable,clip,contextMenu,showBounds)
{
    _socket = socket;
    _updateReady = false;
    _sendReady = false;
    _quit = false;

    _handId = ConfigManager::getInt("value","Plugin.WoundVAC.Hand",0);

    createOrigin();

    int size = 0;
    _socket->recv(&size,sizeof(int));

    std::cerr << "Data Length: " << size << std::endl;

    char * data = new char[size];

    _socket->recv(data,size);

    std::cerr << "Data: " << data << std::endl;

    std::string dataStr = data;

    Json::Reader reader;
    if(!reader.parse(dataStr,_jsonRoot,false))
    {
	std::cerr << "Error Parsing json." << std::endl;
	return;
    }
    std::cerr << "json parsed." << std::endl;

    //_jsonWriterFactory = new Json::StreamWriterBuilder();

    _lineGeode = new osg::Geode();
    addChild(_lineGeode);

    osg::StateSet * stateset = _lineGeode->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
    _lineGeode->setCullingActive(false);

    createDevice();
    createSheet();
    createSponge();

    Message * m = new Message();
    m->type = 21;
    m->data = NULL;
    m->dataSize = 0;

    _messageList.push_back(m);

    startThread();
}

SimulationObject::~SimulationObject()
{
    quit();
    join();
}

void SimulationObject::preFrame()
{
    static float scaleCor = 1000.0;
    static osg::Matrix invCoordCor = osg::Matrix::rotate(osg::Quat(-M_PI/2.0,osg::Vec3(1,0,0)));
    static osg::Matrix invObjCor = osg::Matrix::rotate(osg::Quat(M_PI/2.0,osg::Vec3(1,0,0)));

    _deviceObject->preFrame();
    _sheetObject->preFrame();
    _spongeObject->preFrame();

    _updateLock.lock();

    if(_updateReady)
    {
	updateState();
	_updateReady = false;
    }

    osg::Matrix m = _deviceObject->getTransform();

    //m = m * invCoordCor;

    osg::Vec3 tdevPos = m.getTrans();

    //std::cerr << "DevPos: " << tdevPos.x() << " " << tdevPos.y() << " " << tdevPos.z() << std::endl;

    m = m * osg::Matrix::translate(-tdevPos);

    osg::Quat tdevRot = m.getRotate();

    _devPos.x() = tdevPos.x() / scaleCor;
    _devPos.y() = tdevPos.z() / scaleCor;
    _devPos.z() = tdevPos.y() / scaleCor;

    _devRot.x() = tdevRot.x();
    _devRot.y() = tdevRot.z();
    _devRot.z() = tdevRot.y();
    _devRot.w() = -tdevRot.w();

    
    m = _sheetObject->getTransform();

    //m = m * invCoordCor;

    osg::Vec3 tshPos = m.getTrans();

    m = m * osg::Matrix::translate(-tshPos);

    osg::Quat tshRot = m.getRotate();

    _shPos.x() = tshPos.x() / scaleCor;
    _shPos.y() = tshPos.z() / scaleCor;
    _shPos.z() = tshPos.y() / scaleCor;

    _shRot.x() = tshRot.x();
    _shRot.y() = tshRot.z();
    _shRot.z() = tshRot.y();
    _shRot.w() = -tshRot.w();



    m = _spongeObject->getTransform();

    //m = m * invCoordCor;

    osg::Vec3 tspPos = m.getTrans();

    m = m * osg::Matrix::translate(-tspPos);

    osg::Quat tspRot = m.getRotate();

    _spPos.x() = tspPos.x() / scaleCor;
    _spPos.y() = tspPos.z() / scaleCor;
    _spPos.z() = tspPos.y() / scaleCor;

    _spRot.x() = tspRot.x();
    _spRot.y() = tspRot.z();
    _spRot.z() = tspRot.y();
    _spRot.w() = -tspRot.w();



    m = PluginHelper::getHandMat(_handId) * getWorldToObjectMatrix();

    //m = m * invCoordCor;

    osg::Vec3 thandPos = m.getTrans();

    m = m * osg::Matrix::translate(-thandPos);

    osg::Quat thandRot = m.getRotate();

    _handPos.x() = thandPos.x() / scaleCor;
    _handPos.y() = thandPos.z() / scaleCor;
    _handPos.z() = thandPos.y() / scaleCor;

    _handRot.x() = thandRot.x();
    _handRot.y() = thandRot.z();
    _handRot.z() = thandRot.y();
    _handRot.w() = -thandRot.w();


    _updateLock.unlock();

    updateLines();
}

void SimulationObject::run()
{
    while(1)
    {
	_quitLock.lock();

	if(_quit)
	{
	    _quitLock.unlock();
	    break;
	}

	_quitLock.unlock();

	checkSocket();

	sendMessages();

	// rate limit, 4ms
	struct timespec ts;
	ts.tv_sec = 0;
	ts.tv_nsec = 4000000;
	nanosleep(&ts,NULL);
    }
}

void SimulationObject::quit()
{
    _quitLock.lock();

    _quit = true;

    _quitLock.unlock();
}

void SimulationObject::createOrigin()
{
    osg::Geode * geode = new osg::Geode();

    osg::ShapeDrawable * sd = new osg::ShapeDrawable(new osg::Sphere(osg::Vec3(0,0,0),20));
    sd->setColor(osg::Vec4(0,0,1,0.4));

    geode->addDrawable(sd);

    osg::StateSet * stateset = geode->getOrCreateStateSet();
    stateset->setMode(GL_BLEND,osg::StateAttribute::ON);
    stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);

    addChild(geode);
}

void SimulationObject::createDevice()
{
    static float scaleCor = 1000.0;
    static osg::Quat coordCor(M_PI/2.0,osg::Vec3(1,0,0));
    static osg::Quat objCor(-M_PI/2.0,osg::Vec3(1,0,0));

    float x,y,z,w;
    x = _jsonRoot.get("devPosx",0.0).asFloat() * scaleCor;
    y = _jsonRoot.get("devPosz",0.0).asFloat() * scaleCor;
    z = _jsonRoot.get("devPosy",0.0).asFloat() * scaleCor;

    std::cerr << "Position: " << x << " " << y << " " << z << std::endl;

    osg::Vec3 position(x,y,z);

    x = _jsonRoot.get("devQx",0.0).asFloat();
    y = _jsonRoot.get("devQz",0.0).asFloat();
    z = _jsonRoot.get("devQy",0.0).asFloat();
    w = -_jsonRoot.get("devQw",1.0).asFloat();

    std::cerr << "Rotation: " << x << " " << y << " " << z << " " << w << std::endl;

    osg::Quat q(x,y,z,w);

    _deviceXForm = new osg::MatrixTransform();

    // TODO replace with functional object
    _deviceObject = new ShadowObject("Device",false,true,false,true,false);
    
    osg::Geode * geode = new osg::Geode();

    osg::ShapeDrawable * sd = new osg::ShapeDrawable(new osg::Box(osg::Vec3(0,0,0),100.0));
    sd->setColor(osg::Vec4(1,1,1,0.9));
    geode->addDrawable(sd);

    osg::StateSet * stateset = geode->getOrCreateStateSet();
    stateset->setMode(GL_BLEND,osg::StateAttribute::ON);
    stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);

    osg::MatrixTransform * objOffset = new osg::MatrixTransform();
    osg::Matrix m;
    m.makeTranslate(osg::Vec3(0,0,50));
    objOffset->setMatrix(m);
    objOffset->addChild(geode);

    _deviceObject->addChild(objOffset);
    _deviceXForm->addChild(objOffset);

    addChild(_deviceXForm);
    addChild(_deviceObject);

    m = osg::Matrix::rotate(q) * osg::Matrix::translate(position);// * osg::Matrix::rotate(coordCor);
    _deviceObject->setTransform(m);
    _deviceXForm->setMatrix(m);

    osg::Geometry * geom = new osg::Geometry();
    geom->setUseDisplayList(false);
    geom->setUseVertexBufferObjects(true);

    _deviceLine = new osg::Vec3Array(2);
    osg::Vec4Array * color = new osg::Vec4Array(1);
    color->at(0) = osg::Vec4(1.0,0,0,1.0);

    geom->setVertexArray(_deviceLine);
    geom->setColorArray(color);
    geom->setColorBinding(osg::Geometry::BIND_OVERALL);

    osg::DrawArrays * da = new osg::DrawArrays(osg::PrimitiveSet::LINES,0,_deviceLine->size());
    geom->addPrimitiveSet(da);
    _lineGeode->addDrawable(geom);
}

void SimulationObject::createSheet()
{
    static float scaleCor = 1000.0;
    static osg::Quat coordCor(M_PI/2.0,osg::Vec3(1,0,0));
    static osg::Quat objCor(-M_PI/2.0,osg::Vec3(1,0,0));

    float x,y,z,w;
    x = _jsonRoot.get("shPosx",0.0).asFloat() * scaleCor;
    y = _jsonRoot.get("shPosz",0.0).asFloat() * scaleCor;
    z = _jsonRoot.get("shPosy",0.0).asFloat() * scaleCor;

    std::cerr << "Position: " << x << " " << y << " " << z << std::endl;

    osg::Vec3 position(x,y,z);

    x = _jsonRoot.get("shQx",0.0).asFloat();
    y = _jsonRoot.get("shQz",0.0).asFloat();
    z = _jsonRoot.get("shQy",0.0).asFloat();
    w = -_jsonRoot.get("shQw",1.0).asFloat();

    std::cerr << "Rotation: " << x << " " << y << " " << z << " " << w << std::endl;

    osg::Quat q(x,y,z,w);

    _sheetXForm = new osg::MatrixTransform();

    _sheetObject = new ShadowObject("Sheet",false,true,false,true,false);
    
    osg::Geode * geode = new osg::Geode();

    osg::ShapeDrawable * sd = new osg::ShapeDrawable(new osg::Box(osg::Vec3(0,0,0),100.0));
    sd->setColor(osg::Vec4(1,1,1,0.9));
    geode->addDrawable(sd);

    osg::StateSet * stateset = geode->getOrCreateStateSet();
    stateset->setMode(GL_BLEND,osg::StateAttribute::ON);
    stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);

    osg::MatrixTransform * objOffset = new osg::MatrixTransform();
    osg::Matrix m;
    m.makeTranslate(osg::Vec3(0,0,50));
    objOffset->setMatrix(m);
    objOffset->addChild(geode);

    _sheetObject->addChild(objOffset);
    _sheetXForm->addChild(objOffset);

    addChild(_sheetXForm);
    addChild(_sheetObject);

    m = osg::Matrix::rotate(q) * osg::Matrix::translate(position);// * osg::Matrix::rotate(coordCor);
    _sheetObject->setTransform(m);
    _sheetXForm->setMatrix(m);

    osg::Geometry * geom = new osg::Geometry();
    geom->setUseDisplayList(false);
    geom->setUseVertexBufferObjects(true);

    _sheetLine = new osg::Vec3Array(2);
    osg::Vec4Array * color = new osg::Vec4Array(1);
    color->at(0) = osg::Vec4(1.0,0,0,1.0);

    geom->setVertexArray(_sheetLine);
    geom->setColorArray(color);
    geom->setColorBinding(osg::Geometry::BIND_OVERALL);

    osg::DrawArrays * da = new osg::DrawArrays(osg::PrimitiveSet::LINES,0,_sheetLine->size());
    geom->addPrimitiveSet(da);
    _lineGeode->addDrawable(geom);
}

void SimulationObject::createSponge()
{
    static float scaleCor = 1000.0;
    static osg::Quat coordCor(M_PI/2.0,osg::Vec3(1,0,0));
    static osg::Quat objCor(-M_PI/2.0,osg::Vec3(1,0,0));

    float x,y,z,w;
    x = _jsonRoot.get("spPosx",0.0).asFloat() * scaleCor;
    y = _jsonRoot.get("spPosz",0.0).asFloat() * scaleCor;
    z = _jsonRoot.get("spPosy",0.0).asFloat() * scaleCor;

    std::cerr << "Position: " << x << " " << y << " " << z << std::endl;

    osg::Vec3 position(x,y,z);

    x = _jsonRoot.get("spQx",0.0).asFloat();
    y = _jsonRoot.get("spQz",0.0).asFloat();
    z = _jsonRoot.get("spQy",0.0).asFloat();
    w = -_jsonRoot.get("spQw",1.0).asFloat();

    std::cerr << "Rotation: " << x << " " << y << " " << z << " " << w << std::endl;

    osg::Quat q(x,y,z,w);

    _spongeXForm = new osg::MatrixTransform();

    _spongeObject = new ShadowObject("Sponge",false,true,false,true,false);
    
    osg::Geode * geode = new osg::Geode();

    osg::ShapeDrawable * sd = new osg::ShapeDrawable(new osg::Box(osg::Vec3(0,0,0),100.0));
    sd->setColor(osg::Vec4(1,1,1,0.9));
    geode->addDrawable(sd);

    osg::StateSet * stateset = geode->getOrCreateStateSet();
    stateset->setMode(GL_BLEND,osg::StateAttribute::ON);
    stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);

    osg::MatrixTransform * objOffset = new osg::MatrixTransform();
    osg::Matrix m;
    m.makeTranslate(osg::Vec3(0,0,50));
    objOffset->setMatrix(m);
    objOffset->addChild(geode);

    _spongeObject->addChild(objOffset);
    _spongeXForm->addChild(objOffset);

    addChild(_spongeXForm);
    addChild(_spongeObject);

    m = osg::Matrix::rotate(q) * osg::Matrix::translate(position);// * osg::Matrix::rotate(coordCor);
    _spongeObject->setTransform(m);
    _spongeXForm->setMatrix(m);

    osg::Geometry * geom = new osg::Geometry();
    geom->setUseDisplayList(false);
    geom->setUseVertexBufferObjects(true);

    _spongeLine = new osg::Vec3Array(2);
    osg::Vec4Array * color = new osg::Vec4Array(1);
    color->at(0) = osg::Vec4(1.0,0,0,1.0);

    geom->setVertexArray(_spongeLine);
    geom->setColorArray(color);
    geom->setColorBinding(osg::Geometry::BIND_OVERALL);

    osg::DrawArrays * da = new osg::DrawArrays(osg::PrimitiveSet::LINES,0,_spongeLine->size());
    geom->addPrimitiveSet(da);
    _lineGeode->addDrawable(geom);
}


void SimulationObject::checkSocket()
{
    fd_set readSet;
    struct timeval tv;

    FD_ZERO(&readSet);
    FD_SET(_socket->getSocketFD(), &readSet);
    tv.tv_sec = 0;
    tv.tv_usec = 0;

    if(select(_socket->getSocketFD()+1,&readSet,NULL,NULL,&tv) != -1)
    {
	if(FD_ISSET(_socket->getSocketFD(), &readSet))
	{
	    processMessages();
	}
    }
    else
    {
	std::cerr << "select error" << std::endl;
    }
}

void SimulationObject::processMessages()
{
    int size;

    _socket->recv(&size,sizeof(int));

    //std::cerr << "Total payload size: " << size << std::endl;

    if(size <= 0)
    {
	return;
    }

    char * buffer = new char[size];

    _socket->recv(buffer,size);

    int index = 0;

    size = *((int*)&buffer[index]);
    index += 4;
    while(size > 0)
    {
	char type = buffer[index];
	index++;
	
	//std::cerr << "type: " << (int)type << " size: " << size << std::endl;
	
	switch(type)
	{
	    case 19:
		//std::cerr << "Got remote state req" << std::endl;
		sendUpdate();
		break;
	    case 20:
		//std::cerr << "Got state update" << std::endl;
		{
		    char * data = new char[size];
		    data[size-1] = '\0';
		    strncpy(data,&buffer[index],size-1);
		    std::string dataStr = data;

		    Json::Reader reader;
		    _updateLock.lock();
		    if(!reader.parse(dataStr,_jsonRoot,false))
		    {
			std::cerr << "Error Parsing json." << std::endl;
			std::cerr << "Data: " << dataStr << std::endl;
		    }

		    _updateReady = true;
		    _updateLock.unlock();
		    delete[] data;

		    Message * m = new Message();
		    m->type = 21;
		    m->data = NULL;
		    m->dataSize = 0;

		    _messageLock.lock();
		    _messageList.push_back(m);
		    _messageLock.unlock();
		}
		index = index - 1 + size;
		break;
	    default:
		std::cerr << "Unknown type: " << (int)type << std::endl;
		index = index - 1 + size;
		break;
	}
	size = *((int*)&buffer[index]);
	index += 4;
    }

    delete[] buffer;
}

void SimulationObject::sendMessages()
{
    _messageLock.lock();

    if(!_messageList.size())
    {
	_messageLock.unlock();
	return;
    }

    int totalSize = 0;

    for(int i = 0; i < _messageList.size(); ++i)
    {
	//std::cerr << "Sending message type: " << (int)_messageList[i]->type << std::endl;
	totalSize += sizeof(int) + sizeof(char);
	if(_messageList[i]->data)
	{
	    totalSize += _messageList[i]->dataSize;
	}
    }
    totalSize += sizeof(int);

    int payloadSize = totalSize + sizeof(int);

    char * payload = new char[payloadSize];

    int index = 0;

    *((int*)&payload[index]) = totalSize;
    index += sizeof(int);

    for(int i = 0; i < _messageList.size(); ++i)
    {
	int mySize = sizeof(char);
	if(_messageList[i]->data)
	{
	    mySize += _messageList[i]->dataSize;
	}
	*((int*)&payload[index]) = mySize;
	index += sizeof(int);
	payload[index] = _messageList[i]->type;
	index++;

	if(_messageList[i]->data)
	{
	    memcpy(&payload[index],_messageList[i]->data,_messageList[i]->dataSize);
	    index += _messageList[i]->dataSize;
	}
    }

    *((int*)&payload[index]) = 0;

    for(int i = 0; i < _messageList.size(); ++i)
    {
	if(_messageList[i]->data)
	{
	    delete[] _messageList[i]->data;
	}
	delete _messageList[i];
    }

    _messageList.clear();

    _messageLock.unlock();

    _socket->send(payload,payloadSize);

    delete[] payload;
}

void SimulationObject::sendUpdate()
{
    _updateLock.lock();
    // update json values
    _jsonLocalRoot["devPosx"] = _devPos.x();
    _jsonLocalRoot["devPosy"] = _devPos.y();
    _jsonLocalRoot["devPosz"] = _devPos.z();

    _jsonLocalRoot["devQx"] = _devRot.x();
    _jsonLocalRoot["devQy"] = _devRot.y();
    _jsonLocalRoot["devQz"] = _devRot.z();
    _jsonLocalRoot["devQw"] = _devRot.w();

    _jsonLocalRoot["devGhostOn"] = _deviceObject->isActive();

    _jsonLocalRoot["shPosx"] = _shPos.x();
    _jsonLocalRoot["shPosy"] = _shPos.y();
    _jsonLocalRoot["shPosz"] = _shPos.z();

    _jsonLocalRoot["shQx"] = _shRot.x();
    _jsonLocalRoot["shQy"] = _shRot.y();
    _jsonLocalRoot["shQz"] = _shRot.z();
    _jsonLocalRoot["shQw"] = _shRot.w();

    _jsonLocalRoot["shGhostOn"] = _sheetObject->isActive();

    _jsonLocalRoot["spPosx"] = _spPos.x();
    _jsonLocalRoot["spPosy"] = _spPos.y();
    _jsonLocalRoot["spPosz"] = _spPos.z();

    _jsonLocalRoot["spQx"] = _spRot.x();
    _jsonLocalRoot["spQy"] = _spRot.y();
    _jsonLocalRoot["spQz"] = _spRot.z();
    _jsonLocalRoot["spQw"] = _spRot.w();

    _jsonLocalRoot["spGhostOn"] = _spongeObject->isActive();

    _jsonLocalRoot["handPosx"] = _handPos.x();
    _jsonLocalRoot["handPosy"] = _handPos.y();
    _jsonLocalRoot["handPosz"] = _handPos.z();

    _jsonLocalRoot["handQx"] = _handRot.x();
    _jsonLocalRoot["handQy"] = _handRot.y();
    _jsonLocalRoot["handQz"] = _handRot.z();
    _jsonLocalRoot["handQw"] = _handRot.w();

    _jsonLocalRoot["handOn"] = true;

    _updateLock.unlock();

    std::string update = writeString(_jsonWriterFactory,_jsonLocalRoot);

    Message * m = new Message();
    m->type = 22;
    m->data = new char[update.length()];
    m->dataSize = update.length();
    strncpy(m->data,update.c_str(),update.length());

    _messageLock.lock();

    _messageList.push_back(m);

    _messageLock.unlock();

}

void SimulationObject::updateState()
{
    static float scaleCor = 1000.0;
    static osg::Quat coordCor(M_PI/2.0,osg::Vec3(1,0,0));
    static osg::Quat objCor(-M_PI/2.0,osg::Vec3(1,0,0));

    float x,y,z,w;
    x = _jsonRoot.get("devPosx",0.0).asFloat() * scaleCor;
    y = _jsonRoot.get("devPosz",0.0).asFloat() * scaleCor;
    z = _jsonRoot.get("devPosy",0.0).asFloat() * scaleCor;

    //std::cerr << "Position: " << x << " " << y << " " << z << std::endl;

    osg::Vec3 position(x,y,z);

    x = _jsonRoot.get("devQx",0.0).asFloat();
    y = _jsonRoot.get("devQz",0.0).asFloat();
    z = _jsonRoot.get("devQy",0.0).asFloat();
    w = -_jsonRoot.get("devQw",1.0).asFloat();

    //std::cerr << "Rotation: " << x << " " << y << " " << z << " " << w << std::endl;

    osg::Quat q(x,y,z,w);

    osg::Matrix m = osg::Matrix::rotate(q) * osg::Matrix::translate(position); // * osg::Matrix::rotate(coordCor);
    if(!_deviceObject->isActive())
    {
	_deviceObject->setTransform(m);
    }
    _deviceXForm->setMatrix(m);


    x = _jsonRoot.get("shPosx",0.0).asFloat() * scaleCor;
    y = _jsonRoot.get("shPosz",0.0).asFloat() * scaleCor;
    z = _jsonRoot.get("shPosy",0.0).asFloat() * scaleCor;

    position = osg::Vec3(x,y,z);

    x = _jsonRoot.get("shQx",0.0).asFloat();
    y = _jsonRoot.get("shQz",0.0).asFloat();
    z = _jsonRoot.get("shQy",0.0).asFloat();
    w = -_jsonRoot.get("shQw",1.0).asFloat();

    q = osg::Quat(x,y,z,w);

    m = osg::Matrix::rotate(q) * osg::Matrix::translate(position);// * osg::Matrix::rotate(coordCor);
    if(!_sheetObject->isActive())
    {
	_sheetObject->setTransform(m);
    }
    _sheetXForm->setMatrix(m);


    x = _jsonRoot.get("spPosx",0.0).asFloat() * scaleCor;
    y = _jsonRoot.get("spPosz",0.0).asFloat() * scaleCor;
    z = _jsonRoot.get("spPosy",0.0).asFloat() * scaleCor;

    position = osg::Vec3(x,y,z);

    x = _jsonRoot.get("spQx",0.0).asFloat();
    y = _jsonRoot.get("spQz",0.0).asFloat();
    z = _jsonRoot.get("spQy",0.0).asFloat();
    w = -_jsonRoot.get("spQw",1.0).asFloat();

    q = osg::Quat(x,y,z,w);

    m = osg::Matrix::rotate(q) * osg::Matrix::translate(position);// * osg::Matrix::rotate(coordCor);
    if(!_spongeObject->isActive())
    {
	_spongeObject->setTransform(m);
    }
    _spongeXForm->setMatrix(m);
}

void SimulationObject::updateLines()
{
    osg::Vec3 point;

    _deviceLine->at(0) = point * _deviceObject->getTransform();
    _deviceLine->at(1) = point * _deviceXForm->getMatrix();

    _deviceLine->dirty();

    _sheetLine->at(0) = point * _sheetObject->getTransform();
    _sheetLine->at(1) = point * _sheetXForm->getMatrix();

    _sheetLine->dirty();

    _spongeLine->at(0) = point * _spongeObject->getTransform();
    _spongeLine->at(1) = point * _spongeXForm->getMatrix();

    _spongeLine->dirty();
}
