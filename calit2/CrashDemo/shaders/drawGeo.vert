uniform float frontAlio;
uniform float frontWorld;
uniform float backAlio;
uniform float backWorld;

varying vec3 ls;
varying vec3 norm;

void main()
{
  vec4 color;
  if(gl_Color.g > 0.5)
  {
      gl_Position = ftransform();
      color.r = gl_Color.r;
      color.g = 1.0;
      gl_FrontColor = color;
      gl_BackColor = color;
      return;
  }
  vec4 npos = gl_ModelViewMatrix * gl_Vertex;

  //norm = gl_Normal;

  //vec3 normal = normalize(gl_NormalMatrix * gl_Normal);
  //vec3 normal2 = normalize(gl_NormalMatrix * -gl_Normal);

  //vec3 ptol = normalize(gl_LightSource[0].position.xyz - npos.xyz);
  //gl_TexCoord[1].xyz = normalize(gl_LightSource[0].position.xyz - npos.xyz);
  //gl_FrontSecondaryColor.xyz = normalize(gl_LightSource[0].position.xyz - npos.xyz);
  ls = normalize(gl_LightSource[0].position.xyz - npos.xyz);
  //float diffuse = max(0.0, dot(ptol, normal));
  //float diffuse2 = max(0.0, dot(ptol, normal2));

  color.r = gl_Color.r;
  //color.g = clamp(diffuse+diffuse2, 0.0, 1.0);
  //color.g = 1.0;
  gl_Position = ftransform();

  gl_TexCoord[0]  = gl_Vertex;


  //gl_FrontColor = gl_Color;
  //gl_BackColor = gl_Color;
  gl_FrontColor = color;
  gl_BackColor = color;
}
