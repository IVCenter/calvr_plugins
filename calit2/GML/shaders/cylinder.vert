varying vec3 ls;
varying vec3 direction;

void main()
{
    vec4 npos = gl_ModelViewMatrix * gl_Vertex;
    ls = normalize(gl_LightSource[0].position.xyz - npos.xyz);
    
    direction = -normalize(npos.xyz);

    gl_FrontColor = gl_Color; 
    gl_Position = gl_Vertex;

}
