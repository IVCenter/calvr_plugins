#include "CircularStack.h"

CircularStack::CircularStack(int size)
{
    _size = size;
    _index = 0;

    _sizeList = new int[size];
    _timeList = new float[size];

    for(int i = 0; i < size; i++)
    {
	_sizeList[i] = 0;
	_timeList[i] = 0.0;
    }
}

CircularStack::~CircularStack()
{
    delete[] _sizeList;
    delete[] _timeList;
}

void CircularStack::push(int size, float time)
{
    _sizeList[_index] = size;
    _timeList[_index] = time;
    _index++;
    _index = _index % _size;
}

float CircularStack::getTimePerByte()
{
    int totalSize = 0;
    float totalTime = 0.0;
    for(int i = 0; i < _size; i++)
    {
	totalSize += _sizeList[i];
	totalTime += _timeList[i];
    }
    return totalTime / ((float)totalSize);
}
