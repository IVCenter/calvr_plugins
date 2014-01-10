/* Vertex shader */
varying out float lightIntensity;
 
void main(void)
{
	gl_TexCoord[0] = gl_MultiTexCoord0;

	// Transform normal and position to eye space (for fragment shader)
        vec3 tnorm = normalize( vec3( gl_NormalMatrix * gl_Normal ) );

        vec3 lightPos = normalize(gl_LightSource[0].position.xyz);

        vec4 point = gl_ModelViewMatrix * gl_Vertex;
        lightIntensity = dot( normalize(lightPos - point.xyz), tnorm );
        lightIntensity = abs( lightIntensity );
        lightIntensity *= 1.5;

        gl_Position = gl_ProjectionMatrix * point;
}
