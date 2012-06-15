uniform float frontAlio;
uniform float frontWorld;
uniform float backAlio;
uniform float backWorld;

void main()
{
  vec4 color;
  vec4 npos = gl_ModelViewMatrix * gl_Vertex;

  vec3 normal = normalize(gl_NormalMatrix * gl_Normal);
  vec3 normal2 = normalize(gl_NormalMatrix * -gl_Normal);

  vec3 ptol = normalize(gl_LightSource[0].position.xyz - npos.xyz);
  float diffuse = max(0.0, dot(ptol, normal));
  float diffuse2 = max(0.0, dot(ptol, normal2));

  vec3 gAmb = gl_LightModel.ambient.xyz * gl_FrontMaterial.ambient.xyz;
  //vec3 amb = gl_FrontMaterial.ambient.xyz * gl_LightSource[0].ambient.xyz;
  vec3 amb = gl_LightSource[0].ambient.xyz;
  //color.rgb = clamp(diffuse+diffuse2, 0.0, 1.0) * gl_Color.rgb ;//* gl_FrontMaterial.diffuse.rgb * gl_LightSource[0].diffuse.rgb + gAmb + amb;
  
  color = clamp(diffuse+diffuse2, 0.0, 1.0) * gl_Color;

  gl_Position = ftransform();

  gl_TexCoord[0]  = gl_MultiTexCoord0;

  gl_FrontColor = color;
  gl_BackColor = color;
}
