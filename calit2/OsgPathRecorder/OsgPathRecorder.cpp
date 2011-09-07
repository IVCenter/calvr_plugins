#include "OsgPathRecorder.h"

#include <config/ConfigManager.h>
#include <kernel/PluginHelper.h>
#include <kernel/PluginManager.h>

#include <PluginMessageType.h>

#include <osg/Vec3>

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdio>

CVRPLUGIN(OsgPathRecorder)

OsgPathRecorder::OsgPathRecorder()
{
    //std::cerr << "Sizeof: " << sizeof(osg::Vec3d::value_type) << std::endl;
}

OsgPathRecorder::~OsgPathRecorder()
{
}

bool OsgPathRecorder::init()
{
    _dataDir = ConfigManager::getEntry("Plugin.OsgPathRecorder.DataDir");

    _mode = NONE;
    _timeScale = 1.0;

    _path = new osg::AnimationPath();
    _path->setLoopMode(osg::AnimationPath::NO_LOOPING);

    _prMenu = new SubMenu("OsgPathRecorder");
    _selectMenu = new SubMenu("Select Bin");
    _prMenu->addItem(_selectMenu);

    _activeFile = new MenuText("Active File: None");
    _prMenu->addItem(_activeFile);
    _timeText = new MenuText("Time: 0.0");
    _prMenu->addItem(_timeText);
    _pointsText = new MenuText("Num Points: 0");
    _prMenu->addItem(_pointsText);

    _timeScaleRV = new MenuRangeValue("Speed", 0.1,100,1.0);
    _timeScaleRV->setCallback(this);
    _prMenu->addItem(_timeScaleRV);

    _playbackCB = new MenuCheckbox("Playback Mode", false);
    _playbackCB->setCallback(this);
    _prMenu->addItem(_playbackCB);

    _recordCB = new MenuCheckbox("Record Mode", false);
    _recordCB->setCallback(this);
    _prMenu->addItem(_recordCB);

    _realtimeCB = new MenuCheckbox("Realtime Record",false);
    _realtimeCB->setCallback(this);

    _startB = new MenuButton("Start");
    _startB->setCallback(this);

    _pauseB = new MenuButton("Pause");
    _pauseB->setCallback(this);

    _stopB = new MenuButton("Stop");
    _stopB->setCallback(this);

    _saveB = new MenuButton("Save");
    _saveB->setCallback(this);

    _emitB = new MenuButton("Emit Point");
    _emitB->setCallback(this);

    _gotoFirstB = new MenuButton("Goto First Point");
    _gotoFirstB->setCallback(this);

    _gotoLastB = new MenuButton("Goto Last Point");
    _gotoLastB->setCallback(this);

    _removeLastB = new MenuButton("Remove Last Point");
    _removeLastB->setCallback(this);

    int numfiles = 10;
    for(int i = 0; i < numfiles; i++)
    {
	MenuButton * mb;
	std::stringstream ss;
	ss << "Bin " << i;
	mb = new MenuButton(ss.str());
	mb->setCallback(this);
	_files.push_back(mb);
	_selectMenu->addItem(mb);
    }

    _time = 0;
    _status = STOP;

    PluginHelper::addRootMenuItem(_prMenu);

    return true;
}

void OsgPathRecorder::preFrame()
{
    if(!_mode)
    {
	return;
    }

    if(_mode == PLAYBACK && _status == START)
    {
	if(_path->empty())
	{
	    return;
	}

	_time = _time + (PluginHelper::getLastFrameDuration() * _timeScale);
	if(_time > _path->getLastTime())
	{
	    _time = _path->getLastTime() - 0.0001;
	}

	osg::AnimationPath::AnimationPath::ControlPoint cp;
	_path->getInterpolatedControlPoint(_time,cp);

        /*osg::AnimationPath::TimeControlPointMap::const_iterator first,second;
        second = _path->getTimeControlPointMap().lower_bound(_time);
        
        if(second == _path->getTimeControlPointMap().begin())
        //if(second != _path->getTimeControlPointMap().end())
        {
            for(int i = 0; i < 3; i++)
            {
                ((double*)cp.getPosition().ptr())[i] = second->second.getPosition().ptr()[i];
                ((double*)cp.getScale().ptr())[i] = second->second.getScale().ptr()[i];
            }
            for(int i = 0; i < 4; i++)
            {
                ((double*)cp.getRotation()._v)[i] = second->second.getRotation()._v[i];
            }
        }
        else if(second != _path->getTimeControlPointMap().end())
        {
            first = second;
            --first;
            
            double delta_time = second->first - first->first;
            if(delta_time == 0.0)
            {
                for(int i = 0; i < 3; i++)
                {
                    ((double*)cp.getPosition().ptr())[i] = first->second.getPosition().ptr()[i];
                    ((double*)cp.getScale().ptr())[i] = first->second.getScale().ptr()[i];
                }
                for(int i = 0; i < 4; i++)
                {
                    ((double*)cp.getRotation()._v)[i] = first->second.getRotation()._v[i];
                }
            }
            else
            {
                double ratio = (_time - first->first) / delta_time;
                double one_ratio = ((double)1.0) - ratio;
                std::cerr << "Ratio: " << ratio << std::endl;
                for(int i = 0; i < 3; i++)
                {
                    ((double*)cp.getPosition().ptr())[i] = (one_ratio)*first->second.getPosition().ptr()[i] + ratio*second->second.getPosition().ptr()[i];
                    ((double*)cp.getScale().ptr())[i] = (one_ratio)*first->second.getScale().ptr()[i] + ratio*second->second.getScale().ptr()[i];
                }
                osg::Quat q;
                //q.slerp(ratio,first->second.getRotation(),second->second.getRotation());
                q = (first->second.getRotation()*one_ratio) + (second->second.getRotation()*ratio);
                cp.setRotation(q);
            }
        }
        else
        {
            osg::AnimationPath::AnimationPath::ControlPoint & point = _path->getTimeControlPointMap().rbegin()->second;
            for(int i = 0; i < 3; i++)
            {
                ((double*)cp.getPosition().ptr())[i] = point.getPosition().ptr()[i];
                ((double*)cp.getScale().ptr())[i] = point.getScale().ptr()[i];
            }
            for(int i = 0; i < 4; i++)
            {
                ((double*)cp.getRotation()._v)[i] = point.getRotation()._v[i];
            }
        }*/

        osg::Vec3d myPos = cp.getPosition() * cp.getScale().x();

        osg::Matrix tran;
        tran.makeTranslate(-myPos);

	osg::Matrixd m;
	m.makeRotate(cp.getRotation());
	m.setTrans(-cp.getPosition());

        tran = tran * m;
	PluginHelper::setObjectMatrix(tran);
	PluginHelper::setObjectScale(cp.getScale().x());

	if(_time + 0.0002 > _path->getLastTime())
	{
	    _time = 0;
	    _status = STOP;
	}
    }

    if(_mode == RECORD && _status == START && _realtimeCB->getValue())
    {
	if(!_path->empty())
	{
	    _time = _time + (PluginHelper::getLastFrameDuration() * _timeScale);
	}

	osg::AnimationPath::AnimationPath::ControlPoint cp;
	osg::Vec3d scale;
	double dscale = PluginHelper::getObjectScale();
	scale = osg::Vec3d(dscale,dscale,dscale);
	//osg::Vec3d pos = PluginHelper::getObjectMatrix().getTrans();
	//osg::Quat rot = PluginHelper::getObjectMatrix().getRotate();
        osg::Vec3d pos = PluginHelper::getWorldToObjectTransform().getTrans();
         osg::Quat rot;
         osg::Matrix obj = PluginHelper::getObjectMatrix();
         obj.setTrans(osg::Vec3d(0,0,0));
         rot = obj.getRotate();

	cp.setPosition(pos);
	cp.setRotation(rot);
	cp.setScale(scale);
	_path->insert(_time,cp);
	_lastTransform = PluginHelper::getObjectMatrix();
	_lastScale = dscale;
    }

    if(_mode == RECORD && !_realtimeCB->getValue())
    {
	if(_path->empty())
	{
	    return;
	}

	//osg::Vec3d currentPoint = PluginHelper::getObjectMatrix().getTrans();
	//osg::Vec3d diff = currentPoint - _lastTransform.getTrans();
	osg::Vec3d currentPoint = PluginHelper::getWorldToObjectTransform().getTrans();
	osg::Vec3d diff = currentPoint - _lastPos;
	double dist = fabs(diff.length());

	double disttime = dist * 100.0 * _timeScaleRV->getValue();// / (PluginHelper::getObjectScale());

	_emitTime = _time + disttime;
        std::stringstream timess;
    timess << "Time: " << _emitTime;
    _timeText->setText(timess.str());
    }
    else
    {

    
    std::stringstream timess;
    timess << "Time: " << _time;
    _timeText->setText(timess.str());
    }
}

void OsgPathRecorder::menuCallback(MenuItem * item)
{
    for(int i = 0; i < _files.size(); i++)
    {
	if(item == _files[i])
	{
	    std::stringstream ss;
	    ss << "Active File: Bin " << i;
	    _activeFile->setText(ss.str());
	    _pointsText->setText("Num Points: 0");
	    _timeText->setText("Time: 0.0");
	    _playbackCB->setValue(false);
	    _recordCB->setValue(false);
	    if(_mode == RECORD)
	    {
		_prMenu->removeItem(_realtimeCB);
		_prMenu->removeItem(_startB);
		_prMenu->removeItem(_pauseB);
		_prMenu->removeItem(_stopB);
		_prMenu->removeItem(_saveB);
		_prMenu->removeItem(_emitB);
		_prMenu->removeItem(_gotoLastB);
		_prMenu->removeItem(_gotoFirstB);
		//_prMenu->removeItem(_removeLastB);
	    }
	    else if(_mode == PLAYBACK)
	    {
		_prMenu->removeItem(_startB);
		_prMenu->removeItem(_pauseB);
		_prMenu->removeItem(_stopB);
	    }
	    _time = 0;
	    _path->clear();

	    std::stringstream filess;
	    filess << _dataDir << "/Bin_" << i << ".pth";
	    _currentFile = filess.str();

	    _mode = NONE;
	    _status = STOP;
	    return;
	}
    }

    if(_currentFile.empty())
    {
	return;
    }

    if(item == _recordCB)
    {
	if(_recordCB->getValue())
	{
	    if(_mode == PLAYBACK)
	    {
		_playbackCB->setValue(false);
		_prMenu->removeItem(_startB);
		_prMenu->removeItem(_pauseB);
		_prMenu->removeItem(_stopB);
		_pointsText->setText("Num Points: 0");
		_timeText->setText("Time: 0.0");
		_time = 0;

		_path->clear();
	    }
            _emitTime = _time;
	    _prMenu->addItem(_realtimeCB);
	    _realtimeCB->setValue(false);
	    _prMenu->addItem(_emitB);
	    _prMenu->addItem(_gotoFirstB);
	    _prMenu->addItem(_gotoLastB);
	    //_prMenu->addItem(_removeLastB);
	    _prMenu->addItem(_saveB);

	    _mode = RECORD;
	}
	else
	{
	    _prMenu->removeItem(_realtimeCB);
	    _prMenu->removeItem(_startB);
	    _prMenu->removeItem(_pauseB);
	    _prMenu->removeItem(_stopB);
	    _prMenu->removeItem(_saveB);
	    _prMenu->removeItem(_emitB);
	    _prMenu->removeItem(_gotoLastB);
	    _prMenu->removeItem(_gotoFirstB);
	    //_prMenu->removeItem(_removeLastB);

	    _pointsText->setText("Num Points: 0");
	    _timeText->setText("Time: 0.0");
	    _time = 0;

	    _path->clear();

	    _mode = NONE;
	}
        _numPoints = 0;
	_timeScaleRV->setValue(1.0);
	_timeScale = 1.0;
	_status = STOP;
    }
    else if(item == _playbackCB)
    {
	if(_playbackCB->getValue())
	{
	    if(_mode == RECORD)
	    {
		_recordCB->setValue(false);
		_prMenu->removeItem(_realtimeCB);
		_prMenu->removeItem(_startB);
		_prMenu->removeItem(_pauseB);
		_prMenu->removeItem(_stopB);
		_prMenu->removeItem(_saveB);
		_prMenu->removeItem(_emitB);
		_prMenu->removeItem(_gotoLastB);
		_prMenu->removeItem(_gotoFirstB);
		//_prMenu->removeItem(_removeLastB);

		_pointsText->setText("Num Points: 0");
		_timeText->setText("Time: 0.0");
		_time = 0;

		_path->clear();
	    }

	    loadCurrentFile();

	    _prMenu->addItem(_startB);
	    _prMenu->addItem(_pauseB);
	    _prMenu->addItem(_stopB);

	    _mode = PLAYBACK;
	}
	else
	{
	    _prMenu->removeItem(_startB);
	    _prMenu->removeItem(_pauseB);
	    _prMenu->removeItem(_stopB);

	    _pointsText->setText("Num Points: 0");
	    _timeText->setText("Time: 0.0");
	    _time = 0;

	    _path->clear();

	    _mode = NONE;
	}
	_timeScaleRV->setValue(1.0);
	_timeScale = 1.0;
	_status = STOP;
    }

    if(item == _realtimeCB)
    {
	if(_realtimeCB->getValue())
	{
	    _prMenu->removeItem(_emitB);
	    _prMenu->removeItem(_gotoLastB);
	    _prMenu->removeItem(_gotoFirstB);
	    //_prMenu->removeItem(_removeLastB);
	    _prMenu->removeItem(_saveB);

	    _prMenu->addItem(_startB);
	    _prMenu->addItem(_pauseB);
	    _prMenu->addItem(_stopB);
	    _prMenu->addItem(_saveB);
	}
	else
	{
	    _prMenu->removeItem(_startB);
	    _prMenu->removeItem(_pauseB);
	    _prMenu->removeItem(_stopB);
	    _prMenu->removeItem(_saveB);

	    _prMenu->addItem(_emitB);
	    _prMenu->addItem(_gotoFirstB);
	    _prMenu->addItem(_gotoLastB);
	    //_prMenu->addItem(_removeLastB);
	    _prMenu->addItem(_saveB);
            _emitTime = _time;
	}
        _time = 0;
	_path->clear();
	_status = STOP;
    }

    if(item == _startB)
    {
	if(_mode == RECORD)
	{
	    if(_status != PAUSE)
	    {
		_time = 0;
		_path->clear();
		_pointsText->setText("Num Points: 0");
	    }
	    _status = START;
	}
	else if(_mode == PLAYBACK)
	{
	    _status = START;
	}
    }

    if(item == _pauseB)
    {
	_status = PAUSE;
    }

    if(item == _stopB)
    {
	_status = STOP;
	_time = 0;
    }

    if(item == _saveB)
    {
	saveCurrentPath();
    }

    if(item == _gotoLastB)
    {
	if(!_path->empty())
	{
	    osg::AnimationPath::AnimationPath::ControlPoint cp;
	    _path->getInterpolatedControlPoint(_path->getLastTime(),cp);
	    /*osg::Matrixd m;
	    m.makeRotate(cp.getRotation());
	    m.setTrans(cp.getPosition());
	    PluginHelper::setObjectMatrix(m);
	    PluginHelper::setObjectScale(cp.getScale().x());*/
            osg::Vec3d myPos = cp.getPosition() * cp.getScale().x();

        osg::Matrix tran;
        tran.makeTranslate(-myPos);

        osg::Matrixd m;
        m.makeRotate(cp.getRotation());
        m.setTrans(-cp.getPosition());

        tran = tran * m;
        PluginHelper::setObjectMatrix(tran);
        PluginHelper::setObjectScale(cp.getScale().x());
	}
    }

    if(item == _gotoFirstB)
    {
	if(!_path->empty())
	{
	    osg::AnimationPath::AnimationPath::ControlPoint cp;
	    _path->getInterpolatedControlPoint(_path->getFirstTime(),cp);
	    /*osg::Matrixd m;
	    m.makeRotate(cp.getRotation());
	    m.setTrans(cp.getPosition());
	    PluginHelper::setObjectMatrix(m);
	    PluginHelper::setObjectScale(cp.getScale().x());*/
            osg::Vec3d myPos = cp.getPosition() * cp.getScale().x();

        osg::Matrix tran;
        tran.makeTranslate(-myPos);

        osg::Matrixd m;
        m.makeRotate(cp.getRotation());
        m.setTrans(-cp.getPosition());

        tran = tran * m;
        PluginHelper::setObjectMatrix(tran);
        PluginHelper::setObjectScale(cp.getScale().x());
	}
    }

    if(item == _emitB)
    {
	 osg::AnimationPath::AnimationPath::ControlPoint cp;
	 osg::Vec3d scale;
	 double dscale = PluginHelper::getObjectScale();
	 scale = osg::Vec3d(dscale,dscale,dscale);
	 //osg::Vec3d pos = PluginHelper::getObjectMatrix().getTrans();
	 //osg::Quat rot = PluginHelper::getObjectMatrix().getRotate();
	 //cp.setPosition(pos);
	 //cp.setRotation(rot);
         osg::Vec3d pos = PluginHelper::getWorldToObjectTransform().getTrans();
         /*osg::Vec3d norm = osg::Vec3d(0.0,1.0,0.0);
         norm = norm * PluginHelper::getWorldToObjectTransform();
         norm = norm - pos;
         norm.normalize();

         osg::Quat rot;
         rot.makeRotate(norm,osg::Vec3d(0.0,1.0,0.0));*/
         osg::Quat rot;
         osg::Matrix obj = PluginHelper::getObjectMatrix();
         obj.setTrans(osg::Vec3d(0,0,0));
         rot = obj.getRotate();

         cp.setPosition(pos);
         cp.setRotation(rot);
         
	 cp.setScale(scale);
	 _path->insert(_emitTime,cp);
	 _lastTransform = PluginHelper::getObjectMatrix();
	 _lastPos = PluginHelper::getWorldToObjectTransform().getTrans();
	 _lastScale = dscale;
         _time = _emitTime;
         _numPoints++;

	 std::stringstream pss;
         pss << "Num Points: " << _numPoints;
         _pointsText->setText(pss.str());
    }

    if(_timeScaleRV == item)
    {
	_timeScale = _timeScaleRV->getValue();
    }

}

void OsgPathRecorder::loadCurrentFile()
{
    /*std::ifstream infile;
    infile.open(_currentFile.c_str(), std::ifstream::in);

    if(!infile.fail())
    {
	_path->read(infile);
    }*/

    FILE * infile;
    infile = fopen(_currentFile.c_str(), "rb");
    if(infile != NULL)
    {
	int numEntries;
	if(fread(&numEntries,sizeof(int),1,infile) <= 0)
	{
	    fclose(infile);
	    return;
	}

	osg::AnimationPath::TimeControlPointMap pmap;

	for(int i = 0; i < numEntries; i++)
	{
	    double time;
	    osg::AnimationPath::AnimationPath::ControlPoint cp;

	    fread(&time,sizeof(double),1,infile);
	    fread((void*)cp.getPosition().ptr(),3*sizeof(double),1,infile);
	    fread((void*)cp.getScale().ptr(),3*sizeof(double),1,infile);
	    fread((void*)cp.getRotation()._v,4*sizeof(osg::Quat::value_type),1,infile);

            //std::cerr << "Point " << i << ": time: " << time << " Pos: " << cp.getPosition().x() << " " << cp.getPosition().y() << " " << cp.getPosition().z() << " Scale: " << cp.getScale().x() << std::endl;

	    pmap[time] = cp;
	}

	_path->setTimeControlPointMap(pmap);

	fclose(infile);
    }
}

void OsgPathRecorder::saveCurrentPath()
{
    /*std::ofstream outfile;

    outfile.open(_currentFile.c_str(), std::ofstream::out | std::ofstream::trunc);

    if(!outfile.fail())
    {
	_path->write(outfile);
    }*/

    FILE * outfile;
    outfile = fopen(_currentFile.c_str(), "wb");
    if(outfile != NULL)
    {
	int size = _path->getTimeControlPointMap().size();
	fwrite(&size,sizeof(int),1,outfile);

	for(osg::AnimationPath::TimeControlPointMap::iterator it = _path->getTimeControlPointMap().begin(); it != _path->getTimeControlPointMap().end(); it++)
	{
	    fwrite(&it->first,sizeof(double),1,outfile);
	    fwrite(it->second.getPosition().ptr(),3*sizeof(double),1,outfile);
	    fwrite(it->second.getScale().ptr(),3*sizeof(double),1,outfile);
	    fwrite(it->second.getRotation()._v,4*sizeof(osg::Quat::value_type),1,outfile);
	}

	fclose(outfile);
    }
}

void OsgPathRecorder::message(int type, char * & data, bool)
{
    PathRecorderMessageType mtype = (PathRecorderMessageType)type;
    switch(mtype)
    {
	case PR_SET_RECORD:
	{
	    bool val = *((bool*)data);
	    if(val == _recordCB->getValue())
	    {
		break;
	    }

	    _recordCB->setValue(val);
	    menuCallback(_recordCB);
	    break;
	}
	case PR_SET_PLAYBACK:
	{
	    bool val = *((bool*)data);
	    if(val == _playbackCB->getValue())
	    {
		break;
	    }

	    _playbackCB->setValue(val);
	    menuCallback(_playbackCB);
	    break;
	}
	case PR_SET_ACTIVE_ID:
	{
	    int index = *((int*)data);
	    if(index >= 0 && index < _files.size())
	    {
		menuCallback(_files[index]);
	    }
	    break;
	}
	case PR_SET_PLAYBACK_SPEED:
	{
	    float speed = *((float *)data);
	    _timeScale = speed;
	    break;
	}
	case PR_START:
	{
	    if(_mode == PLAYBACK)
	    {
		menuCallback(_startB);
	    }
	    break;
	}
	case PR_PAUSE:
	{
	    if(_mode == PLAYBACK)
	    {
		menuCallback(_pauseB);
	    }
	    break;
	}
	case PR_STOP:
	{
	    if(_mode == PLAYBACK)
	    {
		menuCallback(_stopB);
	    }
	    break;
	}
	case PR_GET_TIME:
	{
	    PluginManager::instance()->sendMessageByName(data,PR_GET_TIME,(char*)&_time);
	    break;
	}
	// TODO add these
	/*case PR_GET_START_MAT:
	{
	    if(mPathRecordManager->getActiveFileIdx() >= 0)
	    {
		osg::Matrix m;
		float f;
		mPathRecordManager->playbackPathEntry(f, m);

		PluginManager::instance()->sendMessageByName(data,PR_GET_START_MAT,(char*)m.ptr());
	    }
	    break;
	}
	case PR_GET_START_SCALE:
	{
	    if(mPathRecordManager->getActiveFileIdx() >= 0)
	    {
		osg::Matrix m;
		float f;
		mPathRecordManager->playbackPathEntry(f, m);

		PluginManager::instance()->sendMessageByName(data,PR_GET_START_SCALE,(char*)&f);
	    }
	    break;
	}*/
	case PR_IS_STOPPED:
	{
	    bool b = false;
	    if(_status == STOP)
	    {
		b = true;
	    }
	    PluginManager::instance()->sendMessageByName(data,PR_IS_STOPPED,(char*)&b);
	    break;
	}
	default:
	    break;
    }
}
