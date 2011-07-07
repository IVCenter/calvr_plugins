#include "GreenLight.h"

#include <iostream>

bool GreenLight::handleIntersection(osg::Node * iNode)
{
    // Is it one of ours?
    for (int d = 0; d < _door.size(); d++)
        if (_door[d]->nodes.find(iNode) != _door[d]->nodes.end())
        {
            _door[d]->beginAnimation();

            // Handle group animations
            std::list<Entity *>::iterator eit;
            for (eit = _door[d]->group.begin(); eit != _door[d]->group.end(); eit++)
            {
                (*eit)->beginAnimation();
            }

            return true;
        }

    for (int r = 0; r < _rack.size(); r++)
        if (_rack[r]->nodes.find(iNode) != _rack[r]->nodes.end())
        {
            _rack[r]->beginAnimation();

            // Handle group animations
            std::list<Entity *>::iterator eit;
            for (eit = _rack[r]->group.begin(); eit != _rack[r]->group.end(); eit++)
            {
                (*eit)->beginAnimation();
            }

            return true;
        }

    if (_selectHardwareCheckbox->getValue())
    {
        Entity * ent;
        std::map<std::string,Entity*>::iterator mit;
        for (mit = _components.begin(); mit != _components.end(); mit++)
        {
            ent = mit->second;
            if (ent->nodes.find(iNode) != ent->nodes.end())
            {
                if (_selectedEntities.find(ent) != _selectedEntities.end())
                    deselectHardware(ent);
                else
                    selectHardware(ent);
                return true;
            }
        }
    }

    // Not ours
    return false;
}

void GreenLight::selectHardware(Entity * ent)
{
    _selectedEntities.insert(ent);
    ent->setTransparency(false,true);
}

void GreenLight::deselectHardware(Entity * ent)
{
    _selectedEntities.erase(ent);
    ent->setTransparency(true,true);
}
