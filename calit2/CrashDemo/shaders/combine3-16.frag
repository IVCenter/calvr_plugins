uniform sampler2DRect colors[8];
uniform sampler2DRect depth[8];
uniform sampler2DRect depthR8[4];
uniform int textures;
uniform float redLookup[29];
uniform float greenLookup[29];
uniform float blueLookup[29];

void main()
{
    //sampler2DRect text = 0;
    //gl_FragColor = vec4(0.0,0.0,0.0,1.0);
    //gl_FragColor.g = 1.0 - texture2DRect(depth[1],gl_FragCoord.xy).r;
    //gl_FragColor.r = 1.0 - ((texture2DRect(depth[1],gl_FragCoord.xy).r * 16776960 + texture2DRect(depthR8[1],gl_FragCoord.xy).r * 255) / 16777215.0);
    //return;
    //vec4 color = texture2DRect(text,gl_FragCoord.xy);
    //color.r = 0.0;
    //vec4 color;
    //color.g = 0.0;
    //color.b = 0.0;
    //color.a = 1.0;
    //color.r = texture2DRect(depth[1],gl_FragCoord.xy).r;
    //gl_FragColor = color;
    int index = 0;
    vec4 color = texture2DRect(colors[0],gl_FragCoord.xy);
    //gl_FragColor = texture2DRect(colors[0],gl_FragCoord.xy);
    float dpth = 1.0 - texture2DRect(depth[0],gl_FragCoord.xy).r;
    float temp = 1.0 - texture2DRect(depth[1],gl_FragCoord.xy).r;
    /*for(int i = 1; i < textures; i++)
    {
	temp = texture2DRect(depth[i],gl_FragCoord.xy).r;
	if(temp < dpth)
	{
	    temp = dpth;
	    index = i;
	}
    }*/

    if(temp < dpth)
    {
	dpth = temp;
	//gl_FragColor = texture2DRect(colors[1],gl_FragCoord.xy);
	color = texture2DRect(colors[1],gl_FragCoord.xy);
    }

    temp = 1.0 - texture2DRect(depth[2],gl_FragCoord.xy).r;
    if(temp < dpth)
    {
	dpth = temp;
	//gl_FragColor = texture2DRect(colors[2],gl_FragCoord.xy);
	color = texture2DRect(colors[2],gl_FragCoord.xy);
    }

    int colorIndex = int((color.r * 29.0) + 0.15);
    vec3 colorLookup = vec3(redLookup[colorIndex],greenLookup[colorIndex],blueLookup[colorIndex]);
    gl_FragColor.rgb = colorLookup.rgb * color.g;
    //gl_FragColor = texture2DRect(colors[index],gl_FragCoord.xy);
    gl_FragDepth = dpth;
}
