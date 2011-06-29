#include "GreenLight.h"

#include <iostream>

bool GreenLight::handleIntersection(Node * iNode)
{
    // Is it one of ours?
    for (int d = 0; d < _door.size(); d++)
        if (_door[d]->nodes.find(iNode) != _door[d]->nodes.end())
        {
            _door[d]->beginAnimation();

            // Handle group animations
            list<Entity *>::iterator eit;
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
            list<Entity *>::iterator eit;
            for (eit = _rack[r]->group.begin(); eit != _rack[r]->group.end(); eit++)
            {
                (*eit)->beginAnimation();
            }

            return true;
        }

    // Not ours
    return false;
}
