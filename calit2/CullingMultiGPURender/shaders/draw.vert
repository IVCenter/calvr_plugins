/*
 * Vertex shader to apply basic lighting with provided normal
 */

void main()
{
  vec4 color;

  // get point in camera space
  vec4 npos = gl_ModelViewMatrix * gl_Vertex;

  // calculate two sided normals
  vec3 normal = normalize(gl_NormalMatrix * gl_Normal);
  vec3 normal2 = normalize(gl_NormalMatrix * -gl_Normal);

  // get light source direction
  vec3 ptol = normalize(gl_LightSource[0].position.xyz - npos.xyz);

  // calculate diffuse components
  float diffuse = max(0.0, dot(ptol, normal));
  float diffuse2 = max(0.0, dot(ptol, normal2));

  // store color index
  color.r = gl_Color.r;
  // store lighting multiplier
  color.g = clamp(diffuse+diffuse2, 0.0, 1.0);

  // pass values to fragment shader
  gl_Position = ftransform();
  gl_FrontColor = color;
  gl_BackColor = color;
}
