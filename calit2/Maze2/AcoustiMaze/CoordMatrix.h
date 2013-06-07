#ifndef MAZE_COORDMATRIX_H
#define MAZE_COORDMATRIX_H



class CoordMatrix 
{

public:
	double _cx, _cy, _scale;
	double _scx, _scy;
	void set(double x0, double y0, double sc, int width, int height)
    {
		_cx=x0;
		_cy=y0;
		_scale=sc;
		_scx=width/2.0;
		_scy=height/2.0;
    }
	int xi(double tx)
    {
		return (int)((tx-_cx)*_scale+_scx);
    }
	int yi(double ty)
    {
		return (int)(_scy-(ty-_cy)*_scale);
    }
	int sxi(double tsx)
    {
		return (int)(tsx*_scale);
    }
	int syi(double tsy)
    {
		return (int)(tsy*_scale);
    }
	
	double x(double tx)
    {
		return ((tx-_cx)*_scale+_scx);
    }
	double y(double ty)
    {
		return (_scy-(ty-_cy)*_scale);
    }
	double sx(double tsx)
    {
		return (tsx*_scale);
    }
	double sy(double tsy)
    {
		return (tsy*_scale);
    }

};

#endif
