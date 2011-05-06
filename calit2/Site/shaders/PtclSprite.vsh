uniform vec2 pixelsize;
uniform float density;

void main(void)
{

    //check if point should be ignored
    if( gl_Color.a < density)
	return;

    // decrease pixel size with increasing distance to the 
    // eye point
    vec4 worldPos = vec4(gl_Vertex.x,gl_Vertex.y,gl_Vertex.z,1.0);
    vec4 projPos = gl_ModelViewProjectionMatrix * worldPos;

    float dist = projPos.z / projPos.w;
    float distAlpha = (dist+1.0)/2.0;
    gl_PointSize = pixelsize.y - distAlpha * (pixelsize.y - pixelsize.x);

    gl_FrontColor = gl_Color;

    // return projection position
    gl_Position = projPos;
}
