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

    if (_selectionModeCheckbox->getValue())
    {
        Entity * ent;
        std::map<std::string,Entity*>::iterator mit;
        for (mit = _components.begin(); mit != _components.end(); mit++)
        {
            ent = mit->second;
            if (ent->nodes.find(iNode) != ent->nodes.end())
            {
                if (_selectedEntities.find(ent) != _selectedEntities.end())
                    selectHardware(ent,false);
                else
                    selectHardware(ent,true);
                return true;
            }
        }
    }

    // Not ours
    return false;
}

void GreenLight::selectHardware(Entity * ent, bool select)
{
    if (select)
    {
        _selectedEntities.insert(ent);
        ent->setTransparency(false, true);
    }
    else
    {
        _selectedEntities.erase(ent);
        ent->setTransparency(true, true);
    }
}

void GreenLight::selectCluster(std::set< Entity * > * cluster, bool select)
{
    std::set< Entity * >::iterator eit;
    for (eit = cluster->begin(); eit != cluster->end(); eit++)
    {
        bool selected = _selectedEntities.find(*eit) != _selectedEntities.end();
        if (select && !selected)
            selectHardware(*eit, true);
        else if (!select && selected)
            selectHardware(*eit,false);
    }
}
