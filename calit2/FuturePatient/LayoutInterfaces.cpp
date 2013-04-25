#include "LayoutInterfaces.h"

void LayoutLineObject::ref(LayoutTypeObject * object)
{
    if(object)
    {
	_refMap[object] = true;
    }
}

void LayoutLineObject::unref(LayoutTypeObject * object)
{
    if(object)
    {
	_refMap.erase(object);
    }
}

bool LayoutLineObject::hasRef()
{
    return _refMap.size();
}
