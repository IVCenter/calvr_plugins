#include <GL/glew.h>
#include "PanoDrawableLOD.h"

#include <config/ConfigManager.h>
#include <kernel/NodeMask.h>
#include <kernel/ScreenConfig.h>

#include <osg/GraphicsContext>

#include <iostream>
#include <string>
#include <sys/stat.h>
#include <fcntl.h>

std::map<int,std::vector<int> > PanoDrawableLOD::_leftFileIDs;
std::map<int,std::vector<int> > PanoDrawableLOD::_rightFileIDs;
std::map<int,bool> PanoDrawableLOD::_updateDoneMap;
OpenThreads::Mutex PanoDrawableLOD::_initLock;
std::map<int,sph_cache*> PanoDrawableLOD::_cacheMap;
std::map<int,sph_model*> PanoDrawableLOD::_modelMap;

using namespace cvr;

char * loadShaderFile(std::string file)
{
    struct stat st;
    if(stat(file.c_str(),&st) != 0)
    {
        std::cerr << "Error stating shader file: " << file << std::endl;
        return NULL;
    }

    char * fileBuffer;
    int filefd;
    filefd = open(file.c_str(),O_RDONLY);
    if(!filefd)
    {
        std::cerr << "Error opening shader file: " << file << std::endl;
        return NULL;
    }

    fileBuffer = new char[st.st_size+1];
    fileBuffer[st.st_size] = '\0';
    read(filefd,fileBuffer,st.st_size);

    close(filefd);

    return fileBuffer;
}

PanoDrawableLOD::PanoDrawableLOD(std::string leftEyeFile, std::string rightEyeFile, float radius, int mesh, int depth, int size, std::string vertFile, std::string fragFile)
{
    setUseDisplayList(false);
    _badInit = false;
    _leftEyeFiles.push_back(leftEyeFile);
    _rightEyeFiles.push_back(rightEyeFile);
    _radius = radius;
    _mesh = mesh;
    _depth = depth;
    _size = size;
    _currentIndex = 0;

    std::string shaderDir = ConfigManager::getEntry("value","Plugin.PanoViewLOD.ShaderDir","");
    _vertData = loadShaderFile(shaderDir + "/" + vertFile);
    _fragData = loadShaderFile(shaderDir + "/" + fragFile);

    if(!_vertData)
    {
	std::cerr << "Error loading shader file: " << shaderDir + "/" + vertFile << std::endl;
	_badInit = true;
    }
    
    if(!_fragData)
    {
	std::cerr << "Error loading shader file: " << shaderDir + "/" + fragFile << std::endl;
	_badInit = true;
    }

    setUpdateCallback(new PanoUpdate());
}

PanoDrawableLOD::PanoDrawableLOD(std::vector<std::string> & leftEyeFiles, std::vector<std::string> & rightEyeFiles, float radius, int mesh, int depth, int size, std::string vertFile, std::string fragFile)
{
    setUseDisplayList(false);
    _badInit = false;
    _leftEyeFiles = leftEyeFiles;
    _rightEyeFiles = rightEyeFiles;
    _radius = radius;
    _mesh = mesh;
    _depth = depth;
    _size = size;
    _currentIndex = 0;

    std::string shaderDir = ConfigManager::getEntry("value","Plugin.PanoViewLOD.ShaderDir","");
    _vertData = loadShaderFile(shaderDir + "/" + vertFile);
    _fragData = loadShaderFile(shaderDir + "/" + fragFile);

    if(!_vertData)
    {
	std::cerr << "Error loading shader file: " << shaderDir + "/" + vertFile << std::endl;
	_badInit = true;
    }
    
    if(!_fragData)
    {
	std::cerr << "Error loading shader file: " << shaderDir + "/" + fragFile << std::endl;
	_badInit = true;
    }

    setUpdateCallback(new PanoUpdate());
}

PanoDrawableLOD::PanoDrawableLOD(const PanoDrawableLOD&,const osg::CopyOp& copyop)
{
    std::cerr << "PanoDrawableLOD Warning: in copy constructor." << std::endl;
}

PanoDrawableLOD::~PanoDrawableLOD()
{
    if(_vertData)
    {
	delete[] _vertData;
    }
    if(_fragData)
    {
	delete[] _fragData;
    }

    _leftFileIDs.clear();
    _rightFileIDs.clear();
    _updateDoneMap.clear();
    for(std::map<int,sph_cache*>::iterator it = _cacheMap.begin(); it != _cacheMap.end(); it++)
    {
	//make current
	if(it->first < ScreenConfig::instance()->getNumWindows())
	{
	    ScreenConfig::instance()->getWindowInfo(it->first)->gc->makeCurrent();
	}
	delete it->second;
	delete _modelMap[it->first];
    }
    _cacheMap.clear();
    _modelMap.clear();
}

osg::BoundingBox PanoDrawableLOD::computeBound() const
{
    osg::Vec3 size2(_radius, _radius, _radius);
    _boundingBox.init();
    _boundingBox.set(-size2[0], -size2[1], -size2[2], size2[0], size2[1], size2[2]);
    return _boundingBox;
}

void PanoDrawableLOD::updateBoundingBox()
{
    computeBound();
    dirtyBound();
}

void PanoDrawableLOD::drawImplementation(osg::RenderInfo& ri) const
{
    if(_badInit)
    {
	return;
    }

    glPushAttrib(GL_ALL_ATTRIB_BITS);

    int context = ri.getContextID();

    _initLock.lock();

    if(!_cacheMap[context])
    {
        GLenum err = glewInit();
        if (GLEW_OK != err)
        {
            std::cerr << "Error on glew init: " << glewGetErrorString(err) << std::endl;
            _badInit = true;
            _initLock.unlock();
            glPopAttrib();
            return;
        }
	int cachesize = ConfigManager::getInt("value","Plugin.PanoViewLOD.CacheSize",256);
	_cacheMap[context] = new sph_cache(cachesize);
        _cacheMap[context]->set_debug(false);

	_modelMap[context] = new sph_model(*_cacheMap[context],_vertData,_fragData,_mesh,_depth,_size);
	_leftFileIDs[context] = std::vector<int>();
	_rightFileIDs[context] = std::vector<int>();
	for(int i = 0; i < _leftEyeFiles.size(); i++)
	{
	    _leftFileIDs[context].push_back(_cacheMap[context]->add_file(_leftEyeFiles[i]));
	}
	for(int i = 0; i < _rightEyeFiles.size(); i++)
	{
	    _rightFileIDs[context].push_back(_cacheMap[context]->add_file(_rightEyeFiles[i]));
	}
    }

    //TODO: maybe make this only under a context level lock
    if(!_updateDoneMap[context])
    {
	_cacheMap[context]->update(_modelMap[context]->tick());
	_updateDoneMap[context] = true;
    }

    _initLock.unlock();

    bool left = false;

    osg::Node::NodeMask parentMask;

    if(!getNumParents())
    {
	return;
    }

    parentMask = getParent(0)->getNodeMask();

    if((parentMask & CULL_MASK_LEFT) || (parentMask & CULL_MASK))
    {
	left = true;
    }
    else
    {
	left = false;
    }

    osg::Matrix modelview;

    modelview.makeScale(osg::Vec3(_radius,_radius,_radius));
    modelview = modelview * ri.getState()->getModelViewMatrix();

    _modelMap[context]->set_fade(0);
    _modelMap[context]->prep(ri.getState()->getProjectionMatrix().ptr(),modelview.ptr(), (int)ri.getState()->getCurrentViewport()->width(), (int)ri.getState()->getCurrentViewport()->height());
    
    int fileID;
    if(left)
    {
        //std::cerr << "Left" << std::endl;
	if(_currentIndex >= _leftFileIDs[context].size())
	{
	    return;
	}
	fileID = _leftFileIDs[context][_currentIndex];
    }
    else
    {
        //std::cerr << "Right" << std::endl;
	if(_currentIndex >= _rightFileIDs[context].size())
	{
	    return;
	}
	fileID = _rightFileIDs[context][_currentIndex];
    }

    _modelMap[context]->draw(ri.getState()->getProjectionMatrix().ptr(), modelview.ptr(), &fileID, 1, 0, 0);
    glPopAttrib();
}

void PanoDrawableLOD::PanoUpdate::update(osg::NodeVisitor *, osg::Drawable * drawable)
{
    PanoDrawableLOD * pdl = dynamic_cast<PanoDrawableLOD*>(drawable);
    if(!pdl)
    {
	return;
    }

    for(std::map<int,bool>::iterator it = pdl->_updateDoneMap.begin(); it != pdl->_updateDoneMap.end(); it++)
    {
	it->second = false;
    }
}
