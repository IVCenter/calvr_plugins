/*
 * Vertex shader to draw parts with geometry shader lighting
 */

varying vec3 ls;

void main()
{
  vec4 color;

  // get vertex coordinate in camera space
  vec4 npos = gl_ModelViewMatrix * gl_Vertex;

  // calculate normalized light source direction, send to geometry shader
  ls = normalize(gl_LightSource[0].position.xyz - npos.xyz);

  // pass through the color lookup value
  color.r = gl_Color.r;

  // transform the vertex fully
  gl_Position = ftransform();

  // pass vertex to geometry shader in output variable
  gl_TexCoord[0]  = gl_Vertex;


  // set colors
  gl_FrontColor = color;
  gl_BackColor = color;
}
