#ifndef MAZE_MXJ_H
#define MAZE_MXJ_H

#include <AcoustiMaze.h>
#include <string.h>

namespace AcoustiMaze
{

class MXJ_AcoustiMaze// extends MaxObject implements Drawable
{

protected:
    // TODO
	DrawFrame graphicsFrame;
	Graphics2D frameGraphics;
	
    Pose obs, src;
    AcoustiMaze maze;

	std::string dString="void";

	static std::string[] INLET_ASSIST = new std::string[]{
		"inlet 1 help","inl 2 help","i3 help"
	};
	static std::string[] OUTLET_ASSIST = new std::string[]{
		"outlet 1 help"
	};

public:
    // TODO
	MXJ_AcoustiMaze(Atom[] args);
	void redraw(Graphics2D g, CoordMatrix m);
	void anything(std::string message, Atom[] args);
	void list(Atom[] list);


	void bang();
	void inlet(int i);
	void inlet(float f);

};

};

#endif
