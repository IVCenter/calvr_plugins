#include "GreenLight.h"

#include <iostream>

#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/InteractionManager.h>

void GreenLight::doHoverOver(Entity *& last, Entity * current, bool showHover)
{
    const float eScale = 11.0/10.0; // scalar value to expand by
    const float nScale = 10.0/11.0; // scalar value to normalize by

    // fix old as necessary
    if (last != current)
    {
        if (showHover)
        {
            // normalize last hovered over entity, if there is one
            if (last != NULL)
            {
                if (last->asComponent())
                    last->transform->preMult(osg::Matrix::scale(nScale,nScale,nScale));
                else
                    last->defaultColor();
            }

            // expand currently hovered over entity, if there is one
            if (current != NULL)
            {
                if (current->asComponent())
                    current->transform->preMult(osg::Matrix::scale(eScale,eScale,eScale));
                else
                    current->setColor(osg::Vec3(1,1,.5));
            }

            if (current && current->asComponent()){
                _hoverDialog->setText(current->asComponent()->name);
            }else if( current ){
                _hoverDialog->setText("(nothing)");
            }
            else{
                _hoverDialog->setText("(nothing)");
            }
        }

        // assign current to last (notice pass-by-reference)
        last = current;
    }
}

void GreenLight::handleHoverOver(osg::Matrix pointerMat, Entity *& hovered, bool showHover)
{
    osg::Vec3 pointerStart, pointerEnd;
    std::vector<IsectInfo> isecvec;

    pointerStart = pointerMat.getTrans();
    pointerEnd.set(0.0f, 10000.0f, 0.0f);
    pointerEnd = pointerEnd * pointerMat;

    isecvec = getObjectIntersection(cvr::PluginHelper::getScene(),
            pointerStart, pointerEnd);

    if (isecvec.size() == 0)
    {
        doHoverOver(hovered, NULL, showHover);
        return;
    }

    // Optimization
    if (hovered && (hovered->nodes.find(isecvec[0].geode) != hovered->nodes.end()) )
        return;

    // Is it one of ours?
    std::vector<Entity *>::iterator vit;
    for (vit=_door.begin(); vit != _door.end(); vit++)
    {
        if ((*vit)->nodes.find(isecvec[0].geode) != (*vit)->nodes.end())
        {
            doHoverOver(hovered, *vit, showHover);
            return;
        }
    }

    for (vit=_rack.begin(); vit != _rack.end(); vit++)
    {
        if ((*vit)->nodes.find(isecvec[0].geode) != (*vit)->nodes.end())
        {
            doHoverOver(hovered, *vit, showHover);
            return;
        }
    }

    if (_selectionModeCheckbox->getValue())
    {
        std::set< Component * >::iterator sit;
        for (sit = _components.begin(); sit != _components.end(); sit++)
        {
            if ((*sit)->nodes.find(isecvec[0].geode) != (*sit)->nodes.end())
            {
                doHoverOver(hovered, *sit, showHover);
                return;
            }
        }
    }

    // if we get this far, we aren't hovering over anything
    doHoverOver(hovered, NULL, showHover);
}

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

    // Not ours
    return false;
}

bool GreenLight::Component::select(bool select)
{
    if (select != selected)
    {
        // apply new selection and transparency
        selected = select;
        setTransparency(!select);
        return true;
    }
    return false;
}

void GreenLight::selectComponent(Component * comp, bool select)
{
    // only alter if necessary
    if (comp->select(select))
    {
        /* if part of a cluster, change cluster checkbox value as necessary
         * ----------------------------------------------------------------
         * 1)  Only act if the Component is part of a cluster
         * 2)  Find the checkbox that correlates to this cluster
         * 3a) If the checkbox value is true, and this is a deselection, uncheck the box
         * 3b) If the checkbox value is false, and this is a selection,
         *     then check if other nodes in cluster are selected before checking the box
         */

        std::string clusterName = comp->cluster;
        if (clusterName != "")
        {
            std::set< cvr::MenuCheckbox * >::iterator chit;
            for (chit = _clusterCheckbox.begin(); chit != _clusterCheckbox.end(); chit++)
            {
                if ((*chit)->getText() == clusterName)
                     break;
            }

            if (chit == _clusterCheckbox.end())
            {
                std::cerr << "Error: Did not find a checkbox that matches cluster \"" << clusterName << "\"" << std::endl;
                return;
            }

            if ((*chit)->getValue())
            {
                if (!select)
                    (*chit)->setValue(false);
            }
            else if (select) // checkbox value is also false
            {
                std::set< Component * > * clusterSet = _cluster[clusterName];
                std::set< Component * >::iterator sit;
                for (sit = clusterSet->begin(); sit != clusterSet->end(); sit++)
                {
                    if (!(*sit)->selected)
                        return; // should not select checkbox
                }
                // if we get this far, then all components in the cluster are selected
                (*chit)->setValue(true);
            }
        }
    }
}

void GreenLight::selectCluster(std::set< Component * > * cluster, bool select)
{
    std::set< Component * >::iterator sit;
    for (sit = cluster->begin(); sit != cluster->end(); sit++)
    {
            (*sit)->select(select);
    }
}
