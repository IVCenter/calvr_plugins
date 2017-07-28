#ifndef _SENSOR_H
#define _SENSOR_H

#include <string>
#include <vector>
#include <map>
#include <queue>
#include <osg/Uniform>
#include <osg/MatrixTransform>
#include <osg/Drawable>
#include <osgText/Text3D>
#include <osgText/Text>

typedef std::pair<float, float> coord;
	
class FlagTransform : public osg::NodeCallback, public osg::MatrixTransform
{
	public:
    	FlagTransform();
       	virtual void operator()(osg::Node* node, osg::NodeVisitor* nv);
		void setMatrix(osg::Matrix);
		osg::Matrix getMatrix();

	private:
		osg::Matrix _mat;
		OpenThreads::Mutex _mutex;
        bool _changed;
};

class FlagText : public osg::Drawable::UpdateCallback, public osgText::Text3D
{
	public:
    	    FlagText(float size, bool rotate);
       	    virtual void update(osg::NodeVisitor*, osg::Drawable*);
	    void setText(std::string);
	
	protected:
	    FlagText() {};

	private:
	    std::string _text;
	    OpenThreads::Mutex _mutex;
            bool _changed;
};

class Sensor 
{
  	public:
		Sensor(bool type, osg::ref_ptr<osgText::Font> font, osg::ref_ptr<osgText::Style> style, bool rotate = false);
		void setCoord(float lon, float lat) { _location.first = lon; _location.second = lat; }
		void getCoord(double & lon, double & lat) { lon = _location.first; lat = _location.second; }	
		bool getType() { return _type; }	
		float getDirection() { return _direction; }	
		float getVelocity() { return _velocity; }	
		float getTemperature() { return _temperature; }	
		float getPressure() { return _pressure; }	
		float getHumidity() { return _humidity; }
        void setDirection(float direction) { _direction = direction; }	
        void setVelocity(float velocity) { _velocity = velocity; }	
        void setTemperature(float temperature) { _temperature = temperature; }	
        void setPressure(float pressure) { _pressure = pressure; }	
        void setHumidity(float humidity) { _humidity = humidity; }	
		osg::Uniform* getColor() { return _flagcolor; }
		osg::Matrix getRotation() { return _rotation->getMatrix(); }
		void setRotation(osg::Matrix mat) { _rotation->setMatrix(mat); }
		FlagTransform* getFlagTransform() { return _rotation; }
		FlagText* getFlagText() { return _flagText; }
		void setType(bool type) { _type = type; }

	protected:
		Sensor();
        coord _location;
        bool _type;
        float _direction;
        float _velocity;
		float _temperature;
		float _pressure;
		float _humidity;
        FlagTransform* _rotation;
		FlagText* _flagText;
		osg::Uniform* _flagcolor;
};
#endif
